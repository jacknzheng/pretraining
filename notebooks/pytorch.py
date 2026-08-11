class Parameter:
    """
    A simple wrapper mimicking torch.nn.Parameter.
    Holds a value (data) and tracks gradients if needed.
    """
    def __init__(self, data, requires_grad=True):
        self.data = data
        self.requires_grad = requires_grad

    def __repr__(self):
        return f"Parameter(data={self.data}, requires_grad={self.requires_grad})"


class Module:
    """
    A pure Python reconstruction of the core torch.nn.Module base class mechanics.
    Demonstrates implicit registration via __setattr__, sub-module tree traversal, 
    and call hooking lifecycle management.
    """
    def __init__(self):
        # Using alternative storage methods via __dict__ to avoid triggering 
        # infinite loops inside our custom __setattr__ override.
        self.__dict__['_modules'] = {}
        self.__dict__['_parameters'] = {}
        self.__dict__['_buffers'] = {}
        self.__dict__['_training'] = True

    def forward(self, *args, **kwargs):
        """
        To be overridden by subclasses to define the computation lifecycle.
        """
        raise NotImplementedError("Subclasses must implement the forward pass computation.")

    def __call__(self, *args, **kwargs):
        """
        Mimics PyTorch's execution hook dispatch. Intercepts calls to the instance
        to perform pre/post-processing before routing to the custom forward implementation.
        """
        # (In an actual PyTorch module, pre-forward hooks would execute here)
        
        output = self.forward(*args, **kwargs)
        
        # (In an actual PyTorch module, post-forward hooks would execute here)
        return output

    def __setattr__(self, name, value):
        """
        The core 'magic' mechanism of PyTorch. Intercepts property assignments 
        to automatically detect and organize parameters and sub-modules.
        """
        # Safely extract core lookup dicts from state dictionary
        params = self.__dict__.get('_parameters')
        modules = self.__dict__.get('_modules')

        if isinstance(value, Parameter):
            if params is None:
                raise AttributeError("Initialize super().__init__() before assigning parameters.")
            # Remove from sub-modules if it existed there previously
            if modules and name in modules:
                del modules[name]
            params[name] = value
            
        elif isinstance(value, Module):
            if modules is None:
                raise AttributeError("Initialize super().__init__() before assigning sub-modules.")
            # Remove from parameters if it existed there previously
            if params and name in params:
                del params[name]
            modules[name] = value
            
        else:
            # Fall back to standard Python attribute handling for normal variables
            super().__setattr__(name, value)

    def named_parameters(self, memo=None, prefix=''):
        """
        Recursively yields all parameters across the current module and its nested children.
        """
        if memo == None:
            memo = set()
            
        # Yield parameters directly attached to this instance
        for name, param in self._parameters.items():
            if param not in memo:
                memo.add(param)
                full_name = f"{prefix}.{name}" if prefix else name
                yield full_name, param
                
        # Recursively traverse down child sub-modules
        for mname, module in self._modules.items():
            sub_prefix = f"{prefix}.{mname}" if prefix else mname
            yield from module.named_parameters(memo, sub_prefix)

    def parameters(self):
        """
        Returns an iterator over all tracked weights/biases.
        """
        for _, param in self.named_parameters():
            yield param

    def __repr__(self):
        """
        Generates PyTorch's characteristic structured tree string printout.
        """
        lines = []
        for name, module in self._modules.items():
            mod_str = repr(module)
            mod_lines = mod_str.split('\n')
            if len(mod_lines) > 1:
                # Indent child nodes inside multi-line tree layouts
                first_line = f"  ({name}): {mod_lines[0]}"
                rest_lines = [f"    {line}" for line in mod_lines[1:]]
                mod_str = '\n'.join([first_line] + rest_lines)
            else:
                mod_str = f"  ({name}): {mod_str}"
            lines.append(mod_str)
            
        main_str = self.__class__.__name__ + '('
        if lines:
            main_str += '\n' + '\n'.join(lines) + '\n'
        main_str += ')'
        return main_str


# =====================================================================
# Verification and Execution Testing
# =====================================================================

class LinearLayer(Module):
    """A mock implementation of a standard linear transformation layer."""
    def __init__(self, in_features, out_features):
        super().__init__()
        # Simulating matrix weight and bias registration tracking
        self.weight = Parameter(f"Matrix[{out_features}x{in_features}]")
        self.bias = Parameter(f"Vector[{out_features}]")

    def forward(self, x):
        return f"LinearTransform({x})"


class NeuralNetwork(Module):
    """A compound module containing nested sub-modules."""
    def __init__(self):
        super().__init__()
        self.layer1 = LinearLayer(10, 20)
        self.layer2 = LinearLayer(20, 2)
        self.learning_rate = 0.01  # Normal value, ignored by parameter tracker

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


if __name__ == "__main__":
    # 1. Instantiate the compound custom model
    net = NeuralNetwork()

    # 2. Check the structured tree printing format representation
    print("--- Model Structural Tree Layout ---")
    print(net)

    # 3. Verify that named parameter discovery traces nested modules properly
    print("\n--- Tracked Parameters Discovery Log ---")
    for name, param in net.named_parameters():
        print(f"Discovered -> {name}: {param}")

    # 4. Confirm the execution hook lifecyle via functional calls
    print("\n--- Model Execution Test ---")
    output = net("InputTensor")
    print(f"Final Pipeline Return Value: {output}")
