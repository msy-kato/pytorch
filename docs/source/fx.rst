.. currentmodule:: torch.fx

torch.fx
=============

Overview
--------
.. automodule:: torch.fx

.. _Writing Transformations:

Writing Transformations
-----------------------

What is an FX transform? Essentially, it's a function that looks like this.

::

    import torch
    import torch.fx

    def transform(m: nn.Module,
                  tracer_class : type = torch.fx.Tracer) -> torch.nn.Module:
        # Step 1: Acquire a graph representing the code in `m`

        # NOTE: torch.fx.symbolic_trace is a wrapper around a call to
        # fx.Tracer.trace and constructing a GraphModule. We'll
        # split that out in our transform to allow the caller to
        # customize tracing behavior.
        graph : torch.fx.Graph = tracer_class().trace(m)

        # Step 2: Modify this graph or create a new one
        graph = ...

        # Step 3: Construct a Module to return
        return torch.fx.GraphModule(m, graph)

Your transform will take in an :class:`torch.nn.Module`, acquire a :class:`Graph`
from it, do some modifications, and return a new
``nn.Module``. You should think of the ``nn.Module`` that your FX transform
returns as identical to a regular ``nn.Module`` -- you can pass it to another
FX transform, you can pass it to TorchScript, or you can
run it. Ensuring that the inputs and outputs of your FX transform are a
``nn.Module`` will allow for composability.

Given that you’ve passed in an ``nn.Module`` that has been traced into a
graph, there are now two primary approaches you can take to building a new
graph.

Graph Manipulation
^^^^^^^^^^^^^^^^^^

One approach to building this new graph is to simply transform your old
one. To aid in this, we can simply take the graph we obtain from
symbolic tracing and modify it. For example, let’s say we desire to
replace ``torch.add`` with ``torch.mul``.

::

    # Sample module
    class M(torch.nn.Module):
        def forward(self, x, y):
            return torch.add(x, y)

    def transform(m: nn.Module, tracer_class : type = fx.Tracer) -> nn.Module:
        graph : fx.Graph = tracer_class().trace(m)
        # FX represents its graph as an ordered list of
        # nodes, so we can iterate through them.
        for node in graph.nodes:
            # Checks if we're calling a function (i.e:
            # torch.add)
            if node.op == 'call_function':
                # The target attribute is the function
                # that call_function calls.
                if node.target == torch.add:
                    node.target = torch.mul

        graph.lint() # Does some checks to make sure the
                      # graph is well-formed.

        return fx.GraphModule(m, graph)


We can also do more involved graph rewrites, such as deleting or appending
nodes after a node. To aid in these transformations, FX has utility
functions for transforming the graph that can be found in :class:`Graph`. An
example of using these APIs to append a relu can be found below.

::

    with traced.graph.inserting_after(node): # Specifies the
                                             # insertion point
        new_node = traced.graph.call_function(
            torch.relu, args=(node,)) # builds a new relu node
        node.replace_all_uses_with(new_node)

This approach is also a good fit for graph optimizations such as
`conv/batch norm
fusion! <https://github.com/pytorch/pytorch/blob/ec86cec20a8a2312a2295d7bc8be6e88256a2de4/torch/fx/experimental/fuser.py>`__

For simple transformations that only consist of substitutions, you can also
make use of the `subgraph rewriter. <https://github.com/pytorch/pytorch/blob/master/torch/fx/subgraph_rewriter.py>`__

In general, writing your transformation through graph manipulation is a good
fit if you need to make a few small changes or if you need to match multiple
nodes at once. However, if you need to entirely rewrite your graph, you may
want to look at constructing your graph with Proxies (i.e. retracing).

Examples
~~~~~~~~

-  `Replace one
   op <https://github.com/pytorch/pytorch/blob/master/torch/fx/examples/replace_op.py>`__
-  `Conv/Batch Norm
   fusion <https://github.com/pytorch/pytorch/blob/master/torch/fx/experimental/fuser.py>`__
-  `Quantization <https://github.com/pytorch/pytorch/tree/master/torch/quantization/fx>`__

Proxy/Retracing
^^^^^^^^^^^^^^^

Although most transformations can be implemented as graph
transformations, transformations that involve a lot of graph rewrites
are often more easily represented through retracing. For example, let’s
imagine that we wanted to write a pass that decomposed
PyTorch functions. It would transform every ``F.relu(x)``
into ``(x > 0)*x``. One possibility would be to perform the requisite
graph rewriting to insert the comparison and multiplication after the
``F.relu``, and then clean up the original ``F.relu``. However, graph
manipulation can be awkward, and it’s often easier to implicitly
generate the graph by retracing.

To use this method, we write the graph that we want inserted as regular
PyTorch code and pass in Proxy objects. These Proxy objects
will capture the operations that are performed on them and append them to
the graph.

::

    # Note that this decomposition rule can be read as regular Python
    def relu_decomposition(x):
        return (x > 0)*x

    decomposition_rules = {}
    decomposition_rules[F.relu] = relu_decomposition

    def decompose(model: torch.nn.Module, tracer_class : type = fx.Tracer) -> torch.nn.Module:
        graph : fx.Graph = tracer_class().trace(model)
        new_graph = fx.Graph()
        env = {}
        for node in graph.nodes:
            if node.op == 'call_function' and node.target in decomposition_rules:
                # By wrapping the arguments with proxies, we can dispatch to
                # the appropriate decomposition rule and add it to the graph by
                # symbolically tracing it.
                proxy_args = [fx.Proxy(env[x.name]) if isinstance(x, fx.Node) else x for x in node.args]
                new_node = decomposition_rules[node.target](*proxy_args).node
                env[node.name] = new_node
            else:
                new_node = new_graph.node_copy(node, lambda x: env[x.name])
                env[node.name] = new_node
        return fx.GraphModule(model, new_graph)

In addition to avoiding explicit graph manipulation, using Proxies also allows you to
specify your rewrite rules as native Python code. For transformations
that require a large amount of rewrite rules (such as vmap or grad),
this can often improve readability and maintainability of the rules.

TODO: Example transformations (need to be included first)

The Interpreter Pattern
^^^^^^^^^^^^^^^^^^^^^^^

In addition to FX passes that take in a module and return a module,
there may be other things you wish to do with the FX graph. For example,
let’s say that you’d like to obtain
the shape information of tensors in your graph. In this case, instead of
looping over the FX graph and modifying it, you can write an interpreter
on top of the FX graph! As the FX IR is quite simple, it’s easy to
reimplement an interpreter that also captures your desired attributes.

As this pattern is quite useful, we we can also use an abstraction of this pattern
-- the `Interpreter
<https://github.com/pytorch/pytorch/blob/master/torch/fx/interpreter.py>`__.
You can see an example using this for `shape propagation
<https://github.com/pytorch/pytorch/blob/master/torch/fx/passes/shape_prop.py>`__,
which reinterprets the FX graph with example inputs while annotating the
graph with the shapes.

Reinterpreting the FX graph is generally most useful when you want
runtime information that FX typically doesn’t capture (due to being a
symbolic trace). This can be used for capturing shape information for
downstream passes, but it can also be used to capture other information
about execution.
TODO: Add roofline analysis pass once it gets merged.

Examples
~~~~~~~~

-  `Shape
   Propagation <https://github.com/pytorch/pytorch/blob/master/torch/fx/experimental/shape_prop.py>`__
-  `Roofline
   Analyzer <https://github.com/pytorch/pytorch/blob/a9f88511b8155ba9620730fb175dee8c54e346d5/torch/fx/experimental/cost_model.py>`__


Debugging
-----------

Introduction
^^^^^^^^^^^^^^^^

After symbolically tracing an ``nn.Module`` and performing some number
of transformations on the resulting :class:`GraphModule`, we'll want to verify
that the proper semantics were preserved after those transforms. If they
weren't, we may need to do some debugging. The key is to work
backwards: first, check the results of the generated module, then debug
the generated code, then debug the process of transformations that lead
to the generated code.

If you’re not familiar with debuggers, please see the auxiliary section
:ref:`Available Debuggers`.

Debugging the Generated Code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Because FX generates the ``forward()`` function on :class:`GraphModule`\s, using
traditional debugging techniques like ``print`` statements or ``pdb`` is
not as straightfoward. Luckily, we have several techniques we can use
for debugging the generated code.

Use ``pdb``
~~~~~~~~~~~~~
Invoke ``pdb`` to step into the running program. Although the code that
represents the FX graph is not in any source file, we can still step
into it manually using ``pdb`` when the forward pass is invoked.

::

    import torch
    import torch.fx
    import torchvision.models as models

    def my_pass(inp: torch.nn.Module, tracer_class : type = fx.Tracer) -> torch.nn.Module:
        graph = tracer_class().trace(inp)
        # Transformation logic here
        # <...>

        # Return new Module
        return fx.GraphModule(inp, graph)

    my_module = models.resnet18()
    my_module_transformed = my_pass(my_module)

    input_value = torch.randn(5, 3, 224, 224)

    # When this line is executed at runtime, we will be dropped into an
    # interactive `pdb` prompt. We can use the `step` or `s` command to
    # step into the execution of the next line
    import pdb; pdb.set_trace()

    my_module_transformed(input_value)

.. _Print the Generated Code:

Print the Generated Code
~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you’d like to run the same code multiple times, then it can be
a bit tedious to step to the right code with ``pdb``. In that case, one
approach is to simply copy-paste the generated ``forward`` pass into
your code and examine it from there.

::

    # Assume that `traced` is a GraphModule that has undergone some
    # number of transforms

    # Copy this code for later
    print(traced)
    # Print the code generated from symbolic tracing. This outputs:
    """
    def forward(self, y):
        x = self.x
        add_1 = x + y;  x = y = None
        return add_1
    """

    # Subclass the original Module
    class SubclassM(M):
        def __init__(self):
            super().__init__()

        # Paste the generated `forward` function (the one we printed and
        # copied above) here
        def forward(self, y):
            x = self.x
            add_1 = x + y;  x = y = None
            return add_1

    # Create an instance of the original, untraced Module. Then, create an
    # instance of the Module with the copied `forward` function. We can
    # now compare the output of both the original and the traced version.
    pre_trace = M()
    post_trace = SubclassM()

Use the ``to_folder`` Function From ``GraphModule``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:meth:`GraphModule.to_folder` is a method in ``GraphModule`` that allows
you to dump out the generated FX code to a folder. Although copying the
forward pass into the code often suffices as in :ref:`Print the Generated Code`,
it may be easier to examine modules and parameters using ``to_folder``.

::

    m = symbolic_trace(M())
    m.to_folder("foo", "Bar")
    from foo import Bar
    y = Bar()

After running the above example, we can then look at the code within
``foo/module.py`` and modify it as desired (e.g. adding ``print``
statements or using ``pdb``) to debug the generated code.

Debugging the Transformation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now that we've identified that a transformation is creating incorrect
code, it's time to debug the transformation itself. First, we'll check
the :ref:`Limitations of Symbolic Tracing` section in the documentation.
Once we verify that tracing is working as expected, the goal
becomes figuring out what went wrong during our ``GraphModule``
transformation. There may be a quick answer in
:ref:`Writing Transformations`, but, if not, there are several ways to
examine our traced module:

::

    # Sample Module
    class M(torch.nn.Module):
        def forward(self, x, y):
            return x + y

    # Create an instance of `M`
    m = M()

    # Symbolically trace an instance of `M` (returns a GraphModule). In
    # this example, we'll only be discussing how to inspect a
    # GraphModule, so we aren't showing any sample transforms for the
    # sake of brevity.
    traced = symbolic_trace(m)

    # Print the code produced by tracing the module.
    print(traced)
    # The generated `forward` function is:
    """
    def forward(self, x, y):
        add_1 = x + y;  x = y = None
        return add_1
    """

    # Print the internal Graph.
    print(traced.graph)
    # This print-out returns:
    """
    graph(x, y):
        %add_1 : [#users=1] = call_function[target=<built-in function add>](args = (%x, %y), kwargs = {})
        return add_1
    """

    # Print a tabular representation of the internal Graph.
    traced.graph.print_tabular()
    # This gives us:
    """
    opcode         name    target                   args      kwargs
    -------------  ------  -----------------------  --------  --------
    placeholder    x       x                        ()        {}
    placeholder    y       y                        ()        {}
    call_function  add_1   <built-in function add>  (x, y)    {}
    """

Using the utility functions above, we can compare our traced Module
before and after we've applied our transformations. Sometimes, a
simple visual comparison is enough to trace down a bug. If it's still
not clear what's going wrong, a debugger like ``pdb`` can be a good
next step.

Going off of the example above, consider the following code:

::

    # Sample user-defined function
    def transform_graph(module: torch.nn.Module, tracer_class : type = fx.Tracer) -> torch.nn.Module:
        # Get the Graph from our traced Module
        g = tracer_class().trace(module)

        """
        Transformations on `g` go here
        """

        return fx.GraphModule(module, g)

    # Transform the Graph
    transformed = transform_graph(traced)

    # Print the new code after our transforms. Check to see if it was
    # what we expected
    print(transformed)

Using the above example, let’s say that the call to ``print(traced)``
showed us that there was an error in our transforms. We want to find
what goes wrong using a debugger. We start a ``pdb`` session. We can see
what’s happening during the transform by breaking on
``transform_graph(traced)``, then pressing ``s`` to “step into” the call
to ``transform_graph(traced)``.

We may also have good luck by editing the ``print_tabular`` method to print
different attributes of the Nodes in the Graph. (For example, we might
want to see the Node’s ``input_nodes`` and ``users``.)

.. _Available Debuggers:

Available Debuggers
^^^^^^^^^^^^^^^^^^^^^^

The most common Python debugger is
`pdb <https://docs.python.org/3/library/pdb.html>`__. You can start
your program in “debug mode” with ``pdb`` by typing
``python -m pdb FILENAME.py`` into the command line, where ``FILENAME``
is the name of the file you want to debug. After that, you can use the
``pdb`` `debugger commands
<https://docs.python.org/3/library/pdb.html#debugger-commands>`__
to move through your running program stepwise. It’s common to set a
breakpoint (``b LINE-NUMBER``) when you start ``pdb``, then call ``c`` to
run the program until that point. This prevents you from having to step
through each line of execution (using ``s`` or ``n``) to get to the part
of the code you want to examine. Alternatively, you can write
``import pdb; pdb.set_trace()`` before the line you want to break at.
If you add ``pdb.set_trace()``, your program will automatically start
in debug mode when you run it. (In other words, you can just type
``python FILENAME.py`` into the command line instead of
``python -m pdb FILENAME.py``.) Once you're running your file in
debug mode, you can step through the code and examine your program's
internal state using certain commands. There are many excellent
tutorials on ``pdb`` online, including RealPython’s
`“Python Debugging With Pdb” <https://realpython.com/python-debugging-pdb/>`__.

IDEs like PyCharm or VSCode usually have a debugger built in. In your
IDE, you can choose to either a) use ``pdb`` by pulling up a terminal
window in your IDE (e.g. View → Terminal in VSCode), or b) use the
built-in debugger (usually a graphical wrapper around ``pdb``).

.. _Limitations of Symbolic Tracing:

Limitations of Symbolic Tracing
-------------------------------

FX uses a system of **symbolic tracing** (a.k.a `symbolic
execution <https://en.wikipedia.org/wiki/Symbolic_execution>`__)
to capture the semantics of programs in a transformable/analyzable form.
The system is **tracing** in that it executes the program (really an
``nn.Module`` or function) to record operations. It is
**symbolic** in that the data flowing through the program during this
execution is not real data, but rather symbols (:class:`Proxy` in FX parlance).

Although symbolic tracing works for most neural net code, it has some
limitations.

Dynamic Control Flow
^^^^^^^^^^^^^^^^^^^^

The main limitation of symbolic tracing is it does not currently support
*dynamic control flow*. That is, loops or ``if`` statements where the
condition may depend on the input values of the program.

For example, let’s examine the following program:

::

    def func_to_trace(x):
        dim0 = x.size[0]
        if dim0 == 3:
            return torch.relu(x)
        else:
            return torch.neg(x)

    traced = torch.fx.symbolic_trace(func_to_trace)
    """
      <...>
      File "dyn.py", line 6, in func_to_trace
        if dim0 == 3:
      File "pytorch/torch/fx/proxy.py", line 155, in __bool__
        return self.tracer.to_bool(self)
      File "pytorch/torch/fx/proxy.py", line 85, in to_bool
        raise TraceError('symbolically traced variables cannot be used as inputs to control flow')
    torch.fx.proxy.TraceError: symbolically traced variables cannot be used as inputs to control flow
    """

The condition to the ``if`` statement relies on the value of ``dim0``,
which eventually relies on the value of ``x``, a function input. Since
``x`` can change (i.e. if you pass a new input tensor to the traced
function), this is *dynamic control flow*. The traceback walks back up
through your code to show you where this situation happens.

Static Control Flow
~~~~~~~~~~~~~~~~~~~

On the other hand, so-called *static control flow* is supported. Static
control flow is loops or ``if`` statements whose value cannot change
across invocations. Typically, in PyTorch programs, this control flow
arises for code making decisions about a model’s architecture based on
hyper-parameters. As a concrete example:

::

    import torch
    import torch.fx

    class MyModule(torch.nn.Module):
        def __init__(self, do_activation : bool = False):
            super().__init__()
            self.do_activation = do_activation
            self.linear = torch.nn.Linear(512, 512)

        def forward(self, x):
            x = self.linear(x)
            # This if-statement is so-called static control flow.
            # Its condition does not depend on any input values
            if self.do_activation:
                x = torch.relu(x)
            return x

    without_activation = MyModule(do_activation=False)
    with_activation = MyModule(do_activation=True)

    traced_without_activation = torch.fx.symbolic_trace(without_activation)
    print(traced_without_activation.code)
    """
    def forward(self, x):
        linear_1 = self.linear(x);  x = None
        return linear_1
    """

    traced_with_activation = torch.fx.symbolic_trace(with_activation)
    print(traced_with_activation.code)
    """
    import torch
    def forward(self, x):
        linear_1 = self.linear(x);  x = None
        relu_1 = torch.relu(linear_1);  linear_1 = None
        return relu_1
    """

The if-statement ``if self.do_activation`` does not depend on any
function inputs, thus it is static. ``do_activation`` can be considered
to be a hyper-parameter, and the traces of different instances of
``MyModule`` with different values for that parameter have different
code. This is a valid pattern that is supported by symbolic tracing.

Many instances of dynamic control flow are semantically static control
flow. These instances can be made to support symbolic tracing by
removing the data dependencies on input values, for example by moving
values to ``Module`` attributes or by passing constant values during
symbolic tracing:

::

        def f(x, flag):
            if flag: return x
            else: return x*2

        fx.symbolic_trace(f) # Fails!

        def wrapper(flag):
            return lambda x: f(x, flag)

        new_f = wrapper(flag=True)
        fx.symbolic_trace(new_f)

In the case of truly dynamic control flow, the sections of the program
that contain this code can be traced as calls to the Method (see
:ref:`Customizing Tracing`) or function (see
:func:`wrap`) rather than tracing through them.

Non-\ ``torch`` Functions
^^^^^^^^^^^^^^^^^^^^^^^^^

FX uses ``__torch_function__`` as the mechanism by which it intercepts
calls (see the `technical
overview <https://github.com/pytorch/pytorch/blob/master/torch/fx/OVERVIEW.md#technical-details>`__
for more information about this). Some functions, such as builtin Python
functions or those in the ``math`` module, are things that are not
covered by ``__torch_function__``, but we would still like to capture
them in symbolic tracing. For example:

::

    import torch
    import torch.fx
    from math import sqrt

    def normalize(x):
        """
        Normalize `x` by the size of the batch dimension
        """
        return x / sqrt(len(x))

    # It's valid Python code
    normalize(torch.rand(3, 4))

    traced = torch.fx.symbolic_trace(normalize)
    """
      <...>
      File "sqrt.py", line 9, in normalize
        return x / sqrt(len(x))
      File "pytorch/torch/fx/proxy.py", line 161, in __len__
        raise RuntimeError("'len' is not supported in symbolic tracing by default. If you want "
    RuntimeError: 'len' is not supported in symbolic tracing by default. If you want this call to be recorded, please call torch.fx.wrap('len') at module scope
    """

The error tells us that the built-in function ``len`` is not supported.
We can make it so that functions like this are recorded in the trace as
direct calls using the :func:`wrap` API:

::

    torch.fx.wrap('len')
    torch.fx.wrap('sqrt')

    traced = torch.fx.symbolic_trace(normalize)

    print(traced.code)
    """
    import math
    def forward(self, x):
        len_1 = len(x)
        sqrt_1 = math.sqrt(len_1);  len_1 = None
        truediv = x / sqrt_1;  x = sqrt_1 = None
        return truediv
    """

.. _Customizing Tracing:

Customizing Tracing with the ``Tracer`` class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`Tracer` class is the class that underlies the
implementation of ``symbolic_trace``. The behavior of tracing can be
customized by subclassing Tracer, like so:

::

    class MyCustomTracer(torch.fx.Tracer):
        # Inside here you can override various methods
        # to customize tracing. See the `Tracer` API
        # reference
        pass


    # Let's use this custom tracer to trace through this module
    class MyModule(torch.nn.Module):
        def forward(self, x):
            return torch.relu(x) + torch.ones(3, 4)

    mod = MyModule()

    traced_graph = MyCustomTracer().trace(mod)
    # trace() returns a Graph. Let's wrap it up in a
    # GraphModule to make it runnable
    traced = torch.fx.GraphModule(mod, traced_graph)

Leaf Modules
~~~~~~~~~~~~

Leaf Modules are the modules that appear as calls in the symbolic trace
rather than being traced through. The default set of leaf modules is the
set of standard ``torch.nn`` module instances. For example:

::

    class MySpecialSubmodule(torch.nn.Module):
        def forward(self, x):
            return torch.neg(x)

    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 4)
            self.submod = MySpecialSubmodule()

        def forward(self, x):
            return self.submod(self.linear(x))

    traced = torch.fx.symbolic_trace(MyModule())
    print(traced.code)
    # `linear` is preserved as a call, yet `submod` is traced though.
    # This is because the default set of "Leaf Modules" includes all
    # standard `torch.nn` modules.
    """
    import torch
    def forward(self, x):
        linear_1 = self.linear(x);  x = None
        neg_1 = torch.neg(linear_1);  linear_1 = None
        return neg_1
    """

The set of leaf modules can be customized by overriding
:meth:`Tracer.is_leaf_module`.

Miscellanea
^^^^^^^^^^^

-  Tensor constructors (e.g. ``torch.zeros``, ``torch.ones``,
   ``torch.rand``, ``torch.randn``, ``torch.sparse_coo_tensor``)
   are currently not traceable.

   -  The deterministic constructors (``zeros``, ``ones``) can be used
      and the value they produce will be embedded in the trace as a
      constant. This is only problematic if the arguments to these
      constructors refers to dynamic input sizes. In this case,
      ``ones_like`` or ``zeros_like`` may be a viable substitute.
   -  Nondeterministic constructors (``rand``, ``randn``) will have a
      single random value embedded in the trace. This is likely not the
      intended behavior.
   -  This behavior may be fixed in a future release.

-  Type annotations

   -  Python 3-style type annotations (e.g.
      ``func(x : torch.Tensor, y : int) -> torch.Tensor``) are supported
      and will be preserved by symbolic tracing.
   -  Python 2-style comment type annotations
      ``# type: (torch.Tensor, int) -> torch.Tensor`` are not currently
      supported.
   -  Annotations on local names within a function are not currently
      supported.

API Reference
-------------

.. autofunction:: torch.fx.symbolic_trace

.. autofunction:: torch.fx.wrap

.. autoclass:: torch.fx.GraphModule
  :members:

  .. automethod:: __init__

.. autoclass:: torch.fx.Graph
  :members:

  .. automethod:: __init__

.. autoclass:: torch.fx.Node
  :members:

.. autoclass:: torch.fx.Tracer
  :members:

.. autoclass:: torch.fx.Proxy

.. autoclass:: torch.fx.Interpreter
  :members:

.. autoclass:: torch.fx.Transformer
  :members:

.. autofunction:: torch.fx.replace_pattern
