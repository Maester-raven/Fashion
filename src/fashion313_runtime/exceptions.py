class Fashion313RuntimeError(Exception): pass
class InputValidationError(Fashion313RuntimeError): pass
class ParentMaskRequiredError(InputValidationError): pass
class ModelLoadError(Fashion313RuntimeError): pass
class AssetIntegrityError(ModelLoadError): pass
