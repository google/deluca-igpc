from deluca.core import Agent
from deluca.core import field
from deluca.core import Obj
import jax.numpy as jnp


class OpenLoopState(Obj):
    arr: jnp.ndarray = field(jaxed=True)

class OpenLoop(Agent):

    def init(self):
        return OpenLoopState()

    def setup(self):
        self.decay = self.dt / (self.dt + self.RC)

    def __call__(self, state, obs):
        
        action = jax.lax.dynamic_slice(state.h)

        return state.replace(P=P, I=I, D=D), action
