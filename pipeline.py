from typing import Any, Dict, List
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict


@dataclass
class PipelineResult:
    """Result from a single pipeline stage."""
    stage_name: str
    success: bool
    data: Dict[str, Any]
    error: str = None
    
    def to_dict(self):
        return asdict(self)


class PipelineStage(ABC):
    """Base class for pipeline stages."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def process(self, input_data: Any) -> PipelineResult:
        """
        Process input data and return result.
        
        Args:
            input_data: Input to this stage
            
        Returns:
            PipelineResult with output data
        """
        pass


class SmogClassificationPipeline:
    """Extensible pipeline for smog detection and analysis."""
    
    def __init__(self):
        self.stages: List[PipelineStage] = []
        self.results: List[PipelineResult] = []
    
    def add_stage(self, stage: PipelineStage):
        """Add a processing stage to the pipeline."""
        self.stages.append(stage)
        return self
    
    def run(self, image_input: Any) -> Dict[str, Any]:
        """
        Run the entire pipeline on an image.
        
        Args:
            image_input: Image file path or PIL Image
            
        Returns:
            dict with all stage results
        """
        self.results = []
        current_data = {"image_input": image_input}
        
        for stage in self.stages:
            try:
                result = stage.process(current_data)
                self.results.append(result)
                
                if not result.success:
                    break
                
                # Pass output of this stage to next stage
                current_data = result.data
                
            except Exception as e:
                error_result = PipelineResult(
                    stage_name=stage.name,
                    success=False,
                    data={},
                    error=str(e)
                )
                self.results.append(error_result)
                break
        
        # Format output
        return {
            "success": all(r.success for r in self.results),
            "stages": [r.to_dict() for r in self.results],
            "final_data": current_data
        }
    
    def get_result(self, stage_name: str) -> PipelineResult:
        """Get result from a specific stage."""
        for result in self.results:
            if result.stage_name == stage_name:
                return result
        return None
