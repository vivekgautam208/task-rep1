import random


def make_case(seed=1, E=6, P=4, D=4, H=3, top_k=2, windows=4, alias=True, overflow=False, schedule_variant=0, pressure=True, token_reorder=False, rank_relayout=False, dropout=0.2, jitter=0.11):
    rnd=random.Random(seed)
    if P>E:P=E
    experts=[]
    for e in range(E):
        pid=(e%P if alias else min(e,P-1));experts.append({"param_id":pid,"home_rank":rnd.randrange(4)})
    def block(n, step):
        w=[rnd.uniform(-0.55,0.55) for _ in range(n)];m=[rnd.uniform(-0.05,0.05) for _ in range(n)];v=[rnd.uniform(0.002,0.04) for _ in range(n)];ema=[x+rnd.uniform(-.03,.03) for x in w]
        return {"weight":w,"m":m,"v":v,"ema":ema,"step":step,"lr_scale":rnd.choice([0.7,1.0,1.3]),"decay_scale":rnd.choice([0.0,0.5,1.0]),"trainable":(rnd.random()>0.22)}
    router=block(E*(D+1),rnd.randrange(1,6)); router["trainable"]=True; expert_params=[block(H*D+2*H,rnd.randrange(1,7)) for _ in range(P)]
    multipliers=[1.0,0.8,1.15,0.65,1.05]
    cfg={"dim":D,"hidden":H,"top_k":top_k,"capacity_factor":0.55 if pressure else 2.0,"aux_coef":0.07,"z_coef":0.015,"router_jitter":jitter,"expert_dropout":dropout,"rng_seed":seed*1009+17,"max_grad_norm":rnd.choice([0.18,0.32,0.55]),"ema_decay":0.93,"optimizer":{"base_lr":0.011,"beta1":0.84,"beta2":0.94,"eps":1e-8,"weight_decay":0.025}}
    scale=1.0e9 if overflow else rnd.choice([128.0,256.0,512.0]);scaler={"scale":scale,"min_scale":16.0,"max_scale":131072.0,"backoff_factor":0.5,"growth_factor":2.0,"growth_interval":2,"growth_tracker":rnd.randrange(0,2)};scheduler={"multipliers":multipliers,"cursor":rnd.randrange(len(multipliers))}
    ws=[];ordinal=0
    for wi in range(windows):
        mbs=[];mcount=2+(wi+seed)%3
        for mi in range(mcount):
            toks=[];n=3+((seed+wi*3+mi)%5)
            for ti in range(n):
                x=[rnd.uniform(-1.1,1.1) for _ in range(D)];target=rnd.uniform(-0.8,0.8);weight=rnd.choice([0.35,0.7,1.0,1.6,2.2]);mask=(rnd.random()>0.12)
                toks.append({"uid":f"s{seed}-w{wi}-m{mi}-t{ti}","ordinal":ordinal,"rank":rnd.randrange(4),"x":x,"target":target,"weight":weight,"mask":mask});ordinal+=1
            if token_reorder:rnd.shuffle(toks)
            mbs.append({"id":f"m{mi}","tokens":toks})
        mids=[m["id"] for m in mbs]
        if schedule_variant%4==0: events=[*({"op":"forward","microbatch_id":m} for m in mids),*({"op":"backward","microbatch_id":m} for m in reversed(mids)),{"op":"flush"}]
        elif schedule_variant%4==1:
            events=[]
            for m in mids:events.extend([{"op":"forward","microbatch_id":m},{"op":"backward","microbatch_id":m}])
            events.append({"op":"flush"})
        elif schedule_variant%4==2: events=[*({"op":"forward","microbatch_id":m} for m in reversed(mids)),*({"op":"backward","microbatch_id":m} for m in mids),{"op":"flush"}]
        else:
            order=mids[:];rnd.shuffle(order);events=[*({"op":"forward","microbatch_id":m} for m in order),*({"op":"backward","microbatch_id":m} for m in order[1:]+order[:1]),{"op":"flush"}]
        ws.append({"id":f"w{wi}","microbatches":mbs,"events":events})
    if rank_relayout:
        for e in experts:e["home_rank"]=(e["home_rank"]+2)%5
        for w in ws:
            for mb in w["microbatches"]:
                for t in mb["tokens"]:t["rank"]=(int(t["rank"])*3+1)%5
    return {"config":cfg,"router":router,"expert_params":expert_params,"experts":experts,"scheduler":scheduler,"loss_scaler":scaler,"windows":ws}


def make_local_cancel_overflow_case():
    w=make_case(991,E=1,P=1,D=1,H=1,top_k=1,windows=1,alias=False,overflow=False,schedule_variant=1,pressure=False,dropout=0.0,jitter=0.0)
    w["config"]["aux_coef"]=0.0; w["config"]["z_coef"]=0.0; w["config"]["max_grad_norm"]=1.0
    w["loss_scaler"].update(scale=512.0,min_scale=16.0,max_scale=131072.0,backoff_factor=0.5,growth_factor=2.0,growth_interval=2,growth_tracker=0)
    w["router"].update(weight=[0.0,0.0],m=[0.0,0.0],v=[0.01,0.01],ema=[0.0,0.0],step=2,lr_scale=1.0,decay_scale=0.0,trainable=True)
    w["expert_params"][0].update(weight=[0.0,1.0,0.0],m=[0.0,0.0,0.0],v=[0.01,0.01,0.01],ema=[0.0,1.0,0.0],step=2,lr_scale=1.0,decay_scale=0.0,trainable=True)
    m0={"id":"m0","tokens":[{"uid":"cancel-a","ordinal":0,"rank":0,"x":[0.0],"target":200.0,"weight":1.0,"mask":True}]}
    m1={"id":"m1","tokens":[{"uid":"cancel-b","ordinal":1,"rank":3,"x":[0.0],"target":-200.0,"weight":1.0,"mask":True}]}
    w["windows"]=[{"id":"w0","microbatches":[m0,m1],"events":[{"op":"forward","microbatch_id":"m0"},{"op":"backward","microbatch_id":"m0"},{"op":"forward","microbatch_id":"m1"},{"op":"backward","microbatch_id":"m1"},{"op":"flush"}]}]
    return w

def add_retry_events(workload):
    for window in workload["windows"]:
        events=[]
        for i,ev in enumerate(window["events"]):
            events.append(dict(ev))
            if ev["op"] in ("forward","backward") and i%2==0: events.append(dict(ev))
            if ev["op"]=="flush": events.append(dict(ev))
        window["events"]=events
    return workload
