### AMWGJob

type AMWGJob <: GibbsJob
  model::GenericModel
  dpindex::IntegerVector # Indices of dependent variables (parameters and transformations) in model.vertices
  dependent::Vector{Union{Parameter, Transformation}} # Points to model.vertices[dpindex] for faster access
  # logtarget::Function
  sampler::Vector{Union{AMWG, Void}}
  tuner::Vector{Union{RobertsRosenthalMCTuner, Void}}
  range::BasicMCRange
  vstate::VariableStateVector # Vector of variable states ordered according to variables in model.vertices
  dpstate::VariableStateVector # Points to vstate[dpindex] for faster access
  # oldlogtarget::Real
  # newlogtarget::Real
  # sstate::Vector{Union{AMWGState, Void}}
  outopts::Vector # Options related to output
  output::Vector{Union{VariableNState, VariableIOStream, Void}} # Output of model's dependent variables
  ndp::Integer # Number of dependent variables, i.e. length(dependent)
  count::Integer # Current number of post-burnin iterations
  imperative::Bool # If imperative=true then traverse graph imperatively, else declaratively via topological sorting
  plain::Bool # If plain=false then job flow is controlled via tasks, else it is controlled without tasks
  # task::Union{Task, Void}
  verbose::Bool
  # close::Union{Function, Void}
  # resetplain!::Union{Function, Void}
  # reset!::Union{Function, Void}
  # save!::Union{Function, Void}
  # iterate!::Function
  # run!::Function

  function AMWGJob(
    model::GenericModel,
    dpindex::IntegerVector,
    logtarget::Union{Function, Void},
    sampler::Vector,
    tuner::Vector,
    range::BasicMCRange,
    vstate::VariableStateVector,
    outopts::Vector,
    imperative::Bool,
    plain::Bool,
    verbose::Bool,
    check::Bool
  )
    instance = new()

    instance.model = model
    instance.dpindex = dpindex
    instance.sampler = isa(sampler, Vector{Union{AMWG, Void}}) ? sampler : convert(Vector{Union{AMWG, Void}}, sampler)
    instance.tuner = if isa(tuner, Vector{Union{RobertsRosenthalMCTuner, Void}})
        tuner
      else
        convert(Vector{Union{RobertsRosenthalMCTuner, Void}}, tuner)
      end
    instance.vstate = vstate

    if check
      checkin(instance)
    end

    instance.ndp = length(dpindex)

    instance.outopts = isa(outopts, Vector{Dict{Symbol, Any}}) ? outopts : convert(Vector{Dict{Symbol, Any}}, outopts)

    if !imperative
      idxs = [
        model.ofkey[k]
        for k in keys(topological_sort_by_dfs(GenericModel(vertices(model), edges(model), is_directed(model))))
      ]
      instance.dpindex = intersect(idxs, dpindex)

      dpidxs = map(x -> findfirst(dpindex, x), instance.dpindex)

      instance.sampler = instance.sampler[dpidxs]
      instance.tuner = instance.tuner[dpidxs]
      instance.outopts = instance.outopts[dpidxs]
    end

    instance.range = range
    instance.imperative = imperative
    instance.plain = plain
    instance.verbose = verbose

    instance.dependent = instance.model.vertices[instance.dpindex]
    for i in 1:instance.ndp
      instance.dependent[i].states = instance.vstate
    end

    # Stopped here
    instance.logtarget

    instance.dpstate = instance.vstate[instance.dpindex]

    instance.output = Array(Union{VariableNState, VariableIOStream, Void}, instance.ndp)
    for i in 1:instance.ndp
      if isa(instance.dependent[i], Parameter)
        augment_parameter_outopts!(instance.outopts[i])
      else
        augment_variable_outopts!(instance.outopts[i])
      end
      instance.output[i] = initialize_output(instance.dpstate[i], range.npoststeps, instance.outopts[i])
    end

    instance.count = 0

    instance.close = all(o -> o != VariableIOStream, instance.output) ? nothing : eval(codegen(:close, instance))

    nodpjob = all(j -> j == nothing, instance.dpjob)
    instance.resetplain! = nodpjob ? nothing : eval(codegen(:resetplain, instance))

    instance.save! = all(o -> o == nothing, instance.output) ? nothing : eval(codegen(:save, instance))

    instance.iterate! = eval(codegen(:iterate, instance))

    if plain
      instance.task = nothing
      instance.reset! = nodpjob ? nothing : instance.resetplain!
    else
      instance.task = Task(() -> initialize_task!(instance))
      instance.reset! = nodpjob ? nothing : eval(codegen(:resettask, instance))
    end

    instance.run! = eval(codegen(:run, instance))

    instance
  end
end
