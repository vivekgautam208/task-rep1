import math


def all_values(router_grad, expert_grads):
    yield from router_grad
    for grad in expert_grads: yield from grad


def grad_norm(router_grad, expert_grads):
    return math.hypot(*[float(v) for v in all_values(router_grad,expert_grads)])


def adamw(block, grad, optimizer, lr):
    block["step"] = int(block["step"]) + 1
    t=block["step"]; b1=float(optimizer["beta1"]); b2=float(optimizer["beta2"])
    eps=float(optimizer["eps"]); wd=float(optimizer["weight_decay"])*float(block.get("decay_scale",1.0))
    rate=float(lr)*float(block.get("lr_scale",1.0)); c1=1.0-b1**t; c2=1.0-b2**t
    for i,g in enumerate(grad):
        old=float(block["weight"][i]); m=b1*float(block["m"][i])+(1.0-b1)*g; v=b2*float(block["v"][i])+(1.0-b2)*g*g
        block["m"][i]=m; block["v"][i]=v
        block["weight"][i]=old*(1.0-rate*wd)-rate*(m/c1)/(math.sqrt(v/c2)+eps)


def ema(block, decay):
    d=float(decay)
    for i,w in enumerate(block["weight"]): block["ema"][i]=d*float(block["ema"][i])+(1.0-d)*float(w)
