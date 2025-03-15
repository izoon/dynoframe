from typing import Dict, List, Any, Optional, Tuple, Callable
import os
import io
import numpy as np
import fitz  # PyMuPDF
import pdfplumber
import pytesseract
from PIL import Image
import cv2
from ..core.dyno_agent_with_tools import DynoAgentWithTools
from ..dyno_llamaindex import DynoDataLoader
from ..llama_index_compat import (
    Document,
    SimpleDirectoryReader,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
    VectorStoreIndex
)

class PDFProcessingDecisionAgent(DynoAgentWithTools):
    """
    Decision agent that intelligently selects the best PDF processing method based on document characteristics.
    Uses reinforcement learning principles to improve selection over time.
    """
    
    def __init__(self, name="PDFDecisionAgent", 
                 skills=None, 
                 goal="Select optimal PDF processing method",
                 enable_learning=True,
                 sampling_pages=3,
                 input_dependencies=None,
                 tools_dataloaders=None):
        """
        Initialize the PDF Processing Decision Agent.
        
        Args:
            name: Agent name
            skills: Agent skills list
            goal: Agent goal
            enable_learning: Whether to enable learning from past decisions
            sampling_pages: Number of pages to sample for analysis
            input_dependencies: List of input dependencies for processing (e.g., files, objects)
            tools_dataloaders: Dictionary of additional tools or data loaders to use
        """
        if skills is None:
            skills = ["PDF Analysis", "Method Selection", "OCR Evaluation"]
            
        super().__init__(
            name=name,
            role="PDF Processor",
            skills=skills,
            goal=goal,
            enable_learning=enable_learning
        )
        
        self.sampling_pages = sampling_pages
        self.available_methods = ["pymupdf", "pdfplumber", "tesseract"]
        
        # Initialize input dependencies
        self.input_dependencies = input_dependencies if input_dependencies is not None else []
        
        # Initialize tools and data loaders
        self.tools_dataloaders = tools_dataloaders if tools_dataloaders is not None else {}
        
        # Register default tools
        self._register_default_tools()
        
        # Decision history for learning
        self.decision_history = []
        
        # Performance metrics for each method
        self.method_performance = {
            "pymupdf": {"speed": 0.9, "accuracy": 0.8, "table_quality": 0.6, "image_handling": 0.7},
            "pdfplumber": {"speed": 0.7, "accuracy": 0.8, "table_quality": 0.9, "image_handling": 0.6},
            "tesseract": {"speed": 0.5, "accuracy": 0.7, "table_quality": 0.5, "image_handling": 0.9}
        }
        
        # Method weights for different document types (initial values)
        self.method_weights = {
            "digital_text": {"pymupdf": 0.8, "pdfplumber": 0.7, "tesseract": 0.3},
            "scanned": {"pymupdf": 0.3, "pdfplumber": 0.4, "tesseract": 0.9},
            "table_heavy": {"pymupdf": 0.5, "pdfplumber": 0.9, "tesseract": 0.4},
            "image_heavy": {"pymupdf": 0.6, "pdfplumber": 0.5, "tesseract": 0.8},
            "mixed": {"pymupdf": 0.7, "pdfplumber": 0.7, "tesseract": 0.6}
        }
    
    def _register_default_tools(self):
        """Register default tools and data loaders."""
        # Register extraction methods as tools
        self.tools_dataloaders.update({
            "extract_with_pymupdf": self._extract_with_pymupdf,
            "extract_with_pdfplumber": self._extract_with_pdfplumber,
            "extract_with_tesseract": self._extract_with_tesseract
        })
    
    def register_tool(self, name: str, tool_function: Callable) -> None:
        """
        Register a new tool or data loader.
        
        Args:
            name: Name of the tool
            tool_function: Function implementing the tool
        """
        self.tools_dataloaders[name] = tool_function
        self.history.append({
            "task": "Register tool",
            "context": f"Added new tool: {name}",
            "role": self.role
        })
    
    def add_input_dependency(self, dependency: Any) -> None:
        """
        Add a new input dependency.
        
        Args:
            dependency: Input dependency to add
        """
        self.input_dependencies.append(dependency)
        self.history.append({
            "task": "Add input dependency",
            "context": f"Added new dependency: {type(dependency).__name__}",
            "role": self.role
        })
    
    def get_available_tools(self) -> List[str]:
        """Get names of all available tools and data loaders."""
        return list(self.tools_dataloaders.keys())
    
    def use_tool(self, tool_name: str, *args, **kwargs) -> Any:
        """
        Use a registered tool.
        
        Args:
            tool_name: Name of the tool to use
            *args, **kwargs: Arguments to pass to the tool
            
        Returns:
            Result of the tool execution
        """
        if tool_name not in self.tools_dataloaders:
            raise ValueError(f"Tool '{tool_name}' not found. Available tools: {self.get_available_tools()}")
            
        tool = self.tools_dataloaders[tool_name]
        
        self.history.append({
            "task": "Use tool",
            "context": f"Used tool: {tool_name}",
            "role": self.role
        })
        
        return tool(*args, **kwargs)
    
    def analyze_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Analyze PDF characteristics to determine document type.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with document characteristics
        """
        self.history.append({
            "task": "Analyze PDF characteristics",
            "context": pdf_path,
            "role": self.role
        })
        
        try:
            # Open PDF with PyMuPDF for initial analysis
            doc = fitz.open(pdf_path)
            
            page_count = doc.page_count
            sample_pages = min(self.sampling_pages, page_count)
            
            # Initialize counters
            text_content = 0
            image_content = 0
            table_content = 0
            scanned_content = 0
            
            # Sample pages for analysis
            sample_indices = [int(i * page_count / sample_pages) for i in range(sample_pages)]
            
            for page_idx in sample_indices:
                page = doc[page_idx]
                
                # Get text
                text = page.get_text()
                text_length = len(text)
                
                # Get images
                images = page.get_images()
                image_count = len(images)
                
                # Check for tables using heuristics (rectangles, lines)
                rect_count = len(page.search_for(""))  # Approximate table detection
                
                # Calculate text density to detect scanned pages
                if text_length < 100 and image_count > 0:
                    scanned_content += 1
                
                text_content += text_length
                image_content += image_count
                table_content += rect_count
            
            # Normalize by number of sampled pages
            text_content /= sample_pages
            image_content /= sample_pages
            table_content /= sample_pages
            scanned_content /= sample_pages
            
            # Calculate averages and determine document characteristics
            characteristics = {
                "page_count": page_count,
                "text_density": text_content / 1000,  # Per thousand chars
                "image_density": image_content,
                "table_indicators": table_content,
                "scanned_indicators": scanned_content / sample_pages  # Ratio of scanned-looking pages
            }
            
            # Determine document type based on characteristics
            doc_type = self._determine_document_type(characteristics)
            
            doc.close()
            return {
                "characteristics": characteristics,
                "document_type": doc_type
            }
        
        except Exception as e:
            print(f"Error analyzing PDF: {str(e)}")
            return {"error": str(e)}
    
    def _determine_document_type(self, characteristics: Dict[str, float]) -> str:
        """
        Determine document type based on characteristics.
        
        Args:
            characteristics: Dictionary with document characteristics
            
        Returns:
            Document type classification
        """
        # Extract key metrics
        text_density = characteristics["text_density"]
        image_density = characteristics["image_density"]
        table_indicators = characteristics["table_indicators"]
        scanned_indicators = characteristics["scanned_indicators"]
        
        # Decision tree for document type
        if scanned_indicators > 0.6:
            return "scanned"
        elif table_indicators > 5:
            return "table_heavy"
        elif image_density > 2:
            return "image_heavy"
        elif text_density > 2:
            return "digital_text"
        else:
            return "mixed"
    
    def select_method(self, pdf_path: str, extraction_priority: Optional[str] = None) -> Dict[str, Any]:
        """
        Select the best method for processing the PDF.
        
        Args:
            pdf_path: Path to the PDF file
            extraction_priority: Priority type ("speed", "accuracy", "tables", "images")
            
        Returns:
            Dictionary with selected method and rationale
        """
        self.history.append({
            "task": "Select PDF processing method",
            "context": f"{pdf_path} with priority {extraction_priority}",
            "role": self.role
        })
        
        # Analyze PDF
        analysis_result = self.analyze_pdf(pdf_path)
        
        if "error" in analysis_result:
            return {"method": "pymupdf", "rationale": "Error in analysis, using default method"}
        
        doc_type = analysis_result["document_type"]
        characteristics = analysis_result["characteristics"]
        
        # Get method weights for this document type
        weights = self.method_weights[doc_type].copy()
        
        # Adjust weights based on extraction priority
        if extraction_priority:
            if extraction_priority == "speed":
                for method in weights:
                    weights[method] *= self.method_performance[method]["speed"]
            elif extraction_priority == "accuracy":
                for method in weights:
                    weights[method] *= self.method_performance[method]["accuracy"]
            elif extraction_priority == "tables":
                for method in weights:
                    weights[method] *= self.method_performance[method]["table_quality"]
            elif extraction_priority == "images":
                for method in weights:
                    weights[method] *= self.method_performance[method]["image_handling"]
        
        # Select method with highest weight
        selected_method = max(weights, key=weights.get)
        
        # Store decision for learning
        decision = {
            "pdf_path": pdf_path,
            "document_type": doc_type,
            "characteristics": characteristics,
            "selected_method": selected_method,
            "extraction_priority": extraction_priority,
            "success_rating": None  # To be filled after execution
        }
        
        self.decision_history.append(decision)
        
        rationale = self._generate_rationale(selected_method, doc_type, extraction_priority)
        
        return {
            "method": selected_method,
            "document_type": doc_type,
            "rationale": rationale
        }
    
    def _generate_rationale(self, method: str, doc_type: str, priority: Optional[str]) -> str:
        """Generate explanation for method selection."""
        base_rationale = f"Selected {method} for {doc_type} document"
        
        if priority:
            base_rationale += f" with priority on {priority}"
        
        if method == "pymupdf":
            base_rationale += ". PyMuPDF is fast and efficient for general text extraction."
        elif method == "pdfplumber":
            base_rationale += ". PDFPlumber excels at handling tables and structured data."
        elif method == "tesseract":
            base_rationale += ". Tesseract is optimal for OCR on scanned documents or images."
        
        return base_rationale
    
    def process_pdf(self, pdf_path: str, extraction_priority: Optional[str] = None, 
                    use_dependencies: bool = True, custom_method: Optional[str] = None) -> List[Document]:
        """
        Process the PDF using the selected method and convert to LlamaIndex documents.
        
        Args:
            pdf_path: Path to the PDF file
            extraction_priority: Priority type ("speed", "accuracy", "tables", "images")
            use_dependencies: Whether to use registered input dependencies
            custom_method: Force using a specific method instead of auto-selecting
            
        Returns:
            List of LlamaIndex Document objects
        """
        self.history.append({
            "task": "Process PDF",
            "context": pdf_path,
            "role": self.role
        })
        
        # Select method (or use custom method if specified)
        if custom_method:
            if custom_method not in self.available_methods:
                raise ValueError(f"Custom method '{custom_method}' not in available methods: {self.available_methods}")
            method = custom_method
            selection = {"method": method, "document_type": "unknown"}
        else:
            selection = self.select_method(pdf_path, extraction_priority)
            method = selection["method"]
        
        try:
            # Apply input dependencies if available and requested
            preprocessed_path = pdf_path
            if use_dependencies and self.input_dependencies:
                preprocessed_path = self._apply_dependencies(pdf_path)
            
            # Extract text using selected method or custom tool
            tool_name = f"extract_with_{method}" 
            
            if tool_name in self.tools_dataloaders:
                # Use registered tool or default extraction method
                extracted_text = self.use_tool(tool_name, preprocessed_path)
            else:
                # Fallback to standard methods
                if method == "pymupdf":
                    extracted_text = self._extract_with_pymupdf(preprocessed_path)
                elif method == "pdfplumber":
                    extracted_text = self._extract_with_pdfplumber(preprocessed_path)
                elif method == "tesseract":
                    extracted_text = self._extract_with_tesseract(preprocessed_path)
                else:
                    raise ValueError(f"Unknown method: {method}")
            
            # Convert to LlamaIndex Documents
            documents = []
            for i, page_text in enumerate(extracted_text):
                doc = Document(
                    text=page_text,
                    metadata={
                        "source": pdf_path,
                        "page_number": i + 1,
                        "extraction_method": method,
                        "dependencies_applied": use_dependencies and bool(self.input_dependencies)
                    }
                )
                documents.append(doc)
            
            # Add to data loader
            self.data_loader.loaded_data.extend(documents)
            
            # Update success rating for the last decision
            if self.decision_history and not custom_method:
                self.decision_history[-1]["success_rating"] = 1.0  # Assume success
            
            return documents
            
        except Exception as e:
            print(f"Error processing PDF with {method}: {str(e)}")
            
            # Update success rating for the last decision
            if self.decision_history and not custom_method:
                self.decision_history[-1]["success_rating"] = 0.0  # Failure
            
            # Fallback to PyMuPDF if other methods fail
            if method != "pymupdf":
                print(f"Falling back to PyMuPDF")
                return self._extract_with_pymupdf(pdf_path)
            
            return []
    
    def _apply_dependencies(self, pdf_path: str) -> str:
        """
        Apply input dependencies to preprocess the PDF.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Path to the preprocessed PDF (may be the same as input)
        """
        # If no dependencies, return the original path
        if not self.input_dependencies:
            return pdf_path
            
        current_path = pdf_path
        
        # Apply each dependency in sequence
        for dependency in self.input_dependencies:
            # Check if dependency has a process method
            if hasattr(dependency, 'process') and callable(getattr(dependency, 'process')):
                try:
                    # Dependency should return a path to the processed file
                    result = dependency.process(current_path)
                    if result and isinstance(result, str):
                        current_path = result
                except Exception as e:
                    print(f"Error applying dependency {type(dependency).__name__}: {str(e)}")
            
        return current_path
    
    def _extract_with_pymupdf(self, pdf_path: str) -> List[str]:
        """Extract text from PDF using PyMuPDF."""
        doc = fitz.open(pdf_path)
        extracted_text = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            extracted_text.append(text)
        
        doc.close()
        return extracted_text
    
    def _extract_with_pdfplumber(self, pdf_path: str) -> List[str]:
        """Extract text from PDF using PDFPlumber."""
        with pdfplumber.open(pdf_path) as pdf:
            extracted_text = []
            
            for page in pdf.pages:
                text = page.extract_text() or ""
                
                # Extract tables if present
                tables = page.extract_tables()
                table_text = ""
                
                for table in tables:
                    for row in table:
                        table_text += " | ".join([str(cell or "") for cell in row]) + "\n"
                
                combined_text = text + "\n\n" + table_text if table_text else text
                extracted_text.append(combined_text)
            
            return extracted_text
    
    def _extract_with_tesseract(self, pdf_path: str) -> List[str]:
        """Extract text from PDF using PyMuPDF to render pages and Tesseract for OCR."""
        doc = fitz.open(pdf_path)
        extracted_text = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Render page to image
            pix = page.get_pixmap(alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Convert PIL image to OpenCV format
            img_cv = np.array(img)
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
            
            # Preprocess image for better OCR
            img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            img_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
            # OCR with Tesseract
            text = pytesseract.image_to_string(img_thresh)
            extracted_text.append(text)
        
        doc.close()
        return extracted_text
    
    def update_learning(self, pdf_path: str, method: str, success_rating: float) -> None:
        """
        Update learning based on success or failure of a method.
        
        Args:
            pdf_path: Path to the PDF file that was processed
            method: Method that was used
            success_rating: Rating of success (0.0 to 1.0)
        """
        # Find the decision in history
        for decision in self.decision_history:
            if decision["pdf_path"] == pdf_path and decision["selected_method"] == method:
                decision["success_rating"] = success_rating
                
                # Get document type
                doc_type = decision["document_type"]
                
                # Update weights based on success rating
                learning_rate = 0.1
                current_weight = self.method_weights[doc_type][method]
                
                # If success, increase weight; if failure, decrease weight
                if success_rating > 0.5:
                    # Increase weight, but not above 1.0
                    new_weight = min(1.0, current_weight + learning_rate * success_rating)
                else:
                    # Decrease weight, but not below 0.1
                    new_weight = max(0.1, current_weight - learning_rate * (1 - success_rating))
                
                # Update weight
                self.method_weights[doc_type][method] = new_weight
                
                # Normalize weights to ensure they sum close to 1.0
                weight_sum = sum(self.method_weights[doc_type].values())
                for m in self.method_weights[doc_type]:
                    self.method_weights[doc_type][m] /= weight_sum
                
                break
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for learning analysis."""
        method_success = {method: [] for method in self.available_methods}
        doc_type_distribution = {}
        
        for decision in self.decision_history:
            if decision["success_rating"] is not None:
                method = decision["selected_method"]
                method_success[method].append(decision["success_rating"])
                
                doc_type = decision["document_type"]
                doc_type_distribution[doc_type] = doc_type_distribution.get(doc_type, 0) + 1
        
        # Calculate average success rate per method
        avg_success = {}
        for method, ratings in method_success.items():
            if ratings:
                avg_success[method] = sum(ratings) / len(ratings)
            else:
                avg_success[method] = 0.0
        
        return {
            "average_success_by_method": avg_success,
            "document_type_distribution": doc_type_distribution,
            "current_method_weights": self.method_weights,
            "dependencies_count": len(self.input_dependencies),
            "tools_count": len(self.tools_dataloaders),
            "total_decisions": len(self.decision_history)
        }


# Base class for input dependencies
class InputDependency:
    """Base class for input dependencies that can be added to the PDF processing chain."""
    
    def process(self, input_path: str) -> str:
        """
        Process the input and return the path to the processed file.
        
        Args:
            input_path: Path to the input file
            
        Returns:
            Path to the processed file
        """
        raise NotImplementedError("Subclasses must implement process method")


# Example implementations of input dependencies

class ImagePreprocessor(InputDependency):
    """Dependency for preprocessing images in PDFs."""
    
    def __init__(self, contrast_boost=1.5, brightness_boost=1.2):
        self.contrast_boost = contrast_boost
        self.brightness_boost = brightness_boost
    
    def process(self, input_path: str) -> str:
        """Apply contrast and brightness enhancement to PDF pages."""
        # Implementation would extract images, enhance them, and create a new PDF
        # This is a placeholder that would actually modify the PDF
        print(f"ImagePreprocessor: Enhancing images in {input_path}")
        return input_path  # Return the path to the processed file


class OCRPreprocessor(InputDependency):
    """Dependency for preprocessing PDFs for OCR."""
    
    def __init__(self, dpi=300, binarize=True):
        self.dpi = dpi
        self.binarize = binarize
    
    def process(self, input_path: str) -> str:
        """Optimize PDF for OCR processing."""
        # Implementation would optimize the PDF for OCR
        # This is a placeholder that would actually modify the PDF
        print(f"OCRPreprocessor: Optimizing {input_path} for OCR at {self.dpi} DPI")
        return input_path  # Return the path to the processed file


# Example usage
def extract_text_from_pdf(pdf_path: str, extraction_priority: Optional[str] = None) -> List[str]:
    """
    Helper function to extract text from a PDF file using the decision agent.
    
    Args:
        pdf_path: Path to the PDF file
        extraction_priority: Priority for extraction ("speed", "accuracy", "tables", "images")
        
    Returns:
        List of extracted text by page
    """
    # Create agent with dependencies
    image_preprocessor = ImagePreprocessor(contrast_boost=1.8)
    ocr_preprocessor = OCRPreprocessor(dpi=300)
    
    agent = PDFProcessingDecisionAgent()
    
    # Add dependencies
    agent.add_input_dependency(image_preprocessor)
    agent.add_input_dependency(ocr_preprocessor)
    
    # Register custom tools
    agent.register_tool("extract_table_structure", lambda pdf: ["Table structure extraction would happen here"])
    
    # Choose method and process
    selection = agent.select_method(pdf_path, extraction_priority)
    print(f"Selected method: {selection['method']}")
    print(f"Rationale: {selection['rationale']}")
    
    documents = agent.process_pdf(pdf_path, extraction_priority, use_dependencies=True)
    
    # Extract text from documents
    extracted_text = [doc.text for doc in documents]
    
    return extracted_text 