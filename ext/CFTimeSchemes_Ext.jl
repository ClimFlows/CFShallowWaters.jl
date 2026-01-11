module CFTimeSchemes_Ext

import CFTimeSchemes: scratch_space, tendencies!, scratch_space, model_dstate
using CFShallowWaters: CFShallowWaters, AbstractSW

tendencies!(dstate, scratch, model::AbstractSW, state, t) = CFShallowWaters.tendencies!(dstate, scratch, model, state, t)
# scratch_space(model::AbstractSW, state) = CFShallowWaters.scratch_space(model, state)
model_dstate(::AbstractSW, state0) = map(similar, state0)

end
