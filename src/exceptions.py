# src/exceptions.py
"""
Custom exceptions for the data analysis project
"""

class DataAnalysisError(Exception):
    """Base exception for all data analysis errors"""
    pass

class ConfigurationError(DataAnalysisError):
    """Raised when there is a configuration error"""
    def __init__(self, message="Configuration error", details=None):
        self.message = message
        self.details = details
        super().__init__(self.message)

class DataLoadError(DataAnalysisError):
    """Raised when data loading fails"""
    def __init__(self, message="Failed to load data", filepath=None, details=None):
        self.message = message
        self.filepath = filepath
        self.details = details
        super().__init__(self.message)

class DataValidationError(DataAnalysisError):
    """Raised when data validation fails"""
    def __init__(self, message="Data validation failed", column=None, details=None):
        self.message = message
        self.column = column
        self.details = details
        super().__init__(self.message)

class DataCleaningError(DataAnalysisError):
    """Raised when data cleaning fails"""
    def __init__(self, message="Data cleaning failed", details=None):
        self.message = message
        self.details = details
        super().__init__(self.message)

class AnalysisError(DataAnalysisError):
    """Raised when analysis fails"""
    def __init__(self, message="Analysis failed", analysis_type=None, details=None):
        self.message = message
        self.analysis_type = analysis_type
        self.details = details
        super().__init__(self.message)

class ReportGenerationError(DataAnalysisError):
    """Raised when report generation fails"""
    def __init__(self, message="Report generation failed", report_type=None, details=None):
        self.message = message
        self.report_type = report_type
        self.details = details
        super().__init__(self.message)

class VisualizationError(DataAnalysisError):
    """Raised when visualization fails"""
    def __init__(self, message="Visualization failed", plot_type=None, details=None):
        self.message = message
        self.plot_type = plot_type
        self.details = details
        super().__init__(self.message)