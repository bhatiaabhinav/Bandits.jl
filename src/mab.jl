using Random
using MDPs
import MDPs: state_space, action_space, action_meaning, horizon, discount_factor, start_state_probability, transition_probability, reward, is_absorbing, step!, reset!, visualize, state, action, in_absorbing_state

export BernauliMultiArmedBandit

mutable struct BernauliMultiArmedBandit <: AbstractMDP{Int, Int}
    p::Vector{Float64}  # probability of success per arm.

    state::Int
    action::Int
    reward::Float64
    
    BernauliMultiArmedBandit(p) = new(p, 1, 1, 0)
end

@inline action_space(mab::BernauliMultiArmedBandit) = IntegerSpace(length(mab.p))
@inline state_space(::BernauliMultiArmedBandit) = IntegerSpace(2)  # 1 = not pulled yet. 2 = pulled.
action_meaning(::BernauliMultiArmedBandit, a::Int) = "arm $a"
@inline discount_factor(::BernauliMultiArmedBandit) = 0.99
@inline horizon(mab::BernauliMultiArmedBandit) = 1

start_state_probability(mab::BernauliMultiArmedBandit, s::Int) = Float64(s == 1)
transition_probability(mab::BernauliMultiArmedBandit, s::Int, a::Int, s′::Int) = Float64(s′ == 2)

reward(mab::BernauliMultiArmedBandit, s::Int, a::Int, s′::Int) = Float64(s == 1) * mab.p[a]  # expected reward for s,a,s′

is_absorbing(mab::BernauliMultiArmedBandit, s::Int) = s==2


function step!(env::BernauliMultiArmedBandit, a::Int; rng::AbstractRNG=Random.GLOBAL_RNG)::Nothing
    @assert a ∈ action_space(env)
    env.action = a
    if in_absorbing_state(env)
        @warn "The env is already in the absorbing state. Please reset!"
        env.reward = 0
    else
        env.reward = Float64(rand(rng) <= env.p[a])
        env.state = 2
    end    
    nothing
end




