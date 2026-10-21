import math

def all_values(router_grad,expert_grads):
    yield from router_grad
    for g in expert_grads:yield from g

def grad_norm(router_grad,expert_grads):
    # historical implementation adds per-block norms instead of one global L2 norm
    return math.sqrt(sum(v*v for v in router_grad))+sum(math.sqrt(sum(v*v for v in g)) for g in expert_grads)

def adamw(block,grad,opt,lr):
    block["step"]+=1;t=block["step"];b1=float(opt["beta1"]);b2=float(opt["beta2"]);eps=float(opt["eps"]);wd=float(opt["weight_decay"])*float(block.get("decay_scale",1.0));rate=float(lr)*float(block.get("lr_scale",1.0));c1=1-b1**t;c2=1-b2**t
    for i,g in enumerate(grad):
        # coupled decay changes moments
        g=g+wd*block["weight"][i];m=b1*block["m"][i]+(1-b1)*g;v=b2*block["v"][i]+(1-b2)*g*g;block["m"][i]=m;block["v"][i]=v;block["weight"][i]-=rate*(m/c1)/(math.sqrt(v/c2)+eps)
def ema(block,decay):
    for i,w in enumerate(block["weight"]):block["ema"][i]=float(decay)*block["ema"][i]+(1-float(decay))*w
