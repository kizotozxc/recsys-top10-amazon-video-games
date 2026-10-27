import numpy as np
import pandas as pd
import pytest
import torch
from scipy import sparse
from threadpoolctl import threadpool_limits
from models import ItemKNN, ImplicitALS, EASE, FactorizationMachine
from models.custom_als import calibrated_target
from models.fm import FMNetwork
from recsys.features import CausalFeatures, EntityTimeline, make_fm_examples
from recsys.evaluation import Protocol, deterministic_topk, user_metrics, evaluate


@pytest.fixture
def events():
    return pd.DataFrame({'u':[0,0,0,1,1,1,2,2,2], 'i':[0,1,2,0,2,3,1,3,4],
                         'rating':[5.,1.,4.,4.,5.,2.,5.,4.,3.],
                         'timestamp':[10,20,20,10,30,40,10,20,40]})


def test_als_weighted_normal_equations_and_monotonic_loss(events):
    with threadpool_limits(limits=1):
        model=ImplicitALS(factors=3,iterations=6,alpha=5,regularization=2).fit(events,(3,5),50)
        assert np.all(np.diff(model.loss_history) <= 1e-8)
        p=sparse.csr_matrix(((events.rating>=4).astype(float),(events.u,events.i)),shape=(3,5))
        assert p.nnz==len(events)  # retain stored low-rating zero targets
        factors=model._update(p,model.item_factors)
        for u in range(3):
            c=np.ones(5); c[p[u].indices]+=5
            target=p[u].toarray().ravel()
            y=model.item_factors
            lhs=y.T @ (c[:,None]*y) + 2*np.eye(3)
            rhs=y.T @ (c*target)
            np.testing.assert_allclose(lhs@factors[u],rhs,atol=1e-9)


def test_ease_matches_constrained_column_ridge():
    x=sparse.csr_matrix([[1.,1,0],[1,0,1],[0,1,1],[1,1,1]])
    model=EASE(regularization=2).fit(x)
    assert np.all(np.diag(model.weights)==0)
    for j in range(3):
        ix=np.array([k for k in range(3) if k!=j])
        a=x[:,ix].toarray(); y=x[:,j].toarray().ravel()
        expected=np.linalg.solve(a.T@a+2*np.eye(2),a.T@y)
        np.testing.assert_allclose(model.weights[ix,j],expected,atol=1e-10)


def test_knn_cosine_shrinkage_topk():
    x=sparse.csr_matrix([[1.,1,0],[1,0,1],[0,1,1],[1,1,1]])
    model=ItemKNN(neighbours=1,shrinkage=2).fit(x)
    assert np.all(model.weights.diagonal()==0)
    assert np.max(np.diff(model.weights.tocsc().indptr))<=1
    # Every retained pair has overlap 2, norms sqrt(3), shrinkage=2/(2+2).
    np.testing.assert_allclose(model.weights.data,1/3)


def test_causal_features_ignore_current_ties_and_future(events):
    timeline=EntityTimeline(events,'u',3)
    count,total,last,first=timeline.query([0,0,0],[10,20,21])
    np.testing.assert_array_equal(count,[0,1,3])
    np.testing.assert_array_equal(total,[0,5,10])
    features=CausalFeatures(events,(3,5))
    changed=events.copy(); changed.loc[changed.timestamp>=20,'rating']=1
    other=CausalFeatures(changed,(3,5))
    np.testing.assert_array_equal(features.encode([0],[1],[20]),other.encode([0],[1],[20]))


def test_fm_equation_and_fast_score_equal_explicit_pair_sum(events):
    torch.manual_seed(42)
    net=FMNetwork(10,3)
    fields=torch.tensor([[1,3,5,8]])
    explicit=net.bias + net.linear(fields).sum()
    for a in range(4):
        for b in range(a+1,4):
            explicit=explicit+(net.latent(fields[:,a])*net.latent(fields[:,b])).sum()
    torch.testing.assert_close(net(fields)[0],explicit)
    model=FactorizationMachine(factors=3,epochs=2,batch_size=8).fit(events,(3,5),50)
    f=model.features.encode(np.repeat(np.arange(3),5),np.tile(np.arange(5),3),np.full(15,50))
    with torch.no_grad():
        exact=model.network(torch.from_numpy(f)).numpy().reshape(3,5)
    np.testing.assert_allclose(model.score_users(np.arange(3)),exact,atol=2e-7)


def test_custom_signal_is_target_change_and_causal_calibration(events):
    p=calibrated_target(events,50,half_life_days=180)
    baseline=(events.rating.to_numpy()>=4).astype(float)
    assert not np.allclose(p,baseline)
    assert np.all((p>=0)&(p<=1))
    changed=events.copy(); changed.loc[changed.timestamp>20,'rating']=1
    other=calibrated_target(changed,50,half_life_days=180)
    np.testing.assert_allclose(p[events.timestamp<=20],other[events.timestamp<=20])


def test_ranking_ties_and_metric_hand_calculation():
    scores=np.array([[0.,1.,1.,1.,0.]])
    np.testing.assert_array_equal(deterministic_topk(scores,2),[[1,2]])
    recs=np.arange(10)
    ndcg,recall,precision=user_metrics(recs,{0,10},10)
    assert recall==.5 and precision==.1
    assert ndcg==pytest.approx(1/(1+1/np.log2(3)))


def test_temporal_protocol_excludes_cold_and_seen():
    h=pd.DataFrame({'user':['a']*3+['b']*13,
                    'item':['0','1','2']+[str(x) for x in range(13)],
                    'rating':[5.]*16,'timestamp':[1]*16})
    f=pd.DataFrame({'user':['a','a','a','cold'],'item':['3','0','new','4'],
                    'rating':[5.,5.,5.,5.],'timestamp':[3]*4})
    protocol=Protocol(h,f,2,'unit',min_positive_history=1)
    assert len(protocol.users)==1
    assert protocol.truth[protocol.users[0]]=={int(np.flatnonzero(protocol.item_ids=='3')[0])}
    assert protocol.audit['cold_user_events']==1
    assert protocol.audit['cold_item_events']==1
    model=ItemKNN().fit(protocol.positive)
    metrics, users, recs=evaluate(model,protocol)
    assert len(recs[0])==10
    assert not set(recs[0]).intersection(protocol.seen[protocol.users[0]].indices)


def test_negative_sampling_cannot_use_future_catalog_or_past_seen_items(events):
    u,i,t,y=make_fm_examples(events,(3,5),seed=7,negatives_per_positive=3)
    observed={(int(r.u),int(r.i),int(r.timestamp)) for r in events.itertuples()}
    first=events.groupby('i').timestamp.min().to_dict()
    for user,item,time,label in zip(u,i,t,y):
        if (int(user),int(item),int(time)) in observed:
            continue
        assert label==0
        assert first[item] < time
        seen=events[(events.u==user)&(events.timestamp<=time)].i
        assert item not in set(seen)
