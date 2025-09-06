"""Mock device module for Sphinx documentation."""

# Mock DEVICE constant to avoid circular imports
DEVICE = "cpu"

class MockDevice:
    def __str__(self):
        return "cpu"
    
    def __repr__(self):
        return "device(type='cpu')"

# Create a mock device instance
device = MockDevice()