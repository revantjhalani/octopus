import json
import google.generativeai as genai
import time
import os
from typing import Dict, List, Any
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from functools import partial

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProductEnhancer:
    def __init__(self, api_key: str, max_workers: int = 5, requests_per_second: float = 2.0):
        """Initialize the ProductEnhancer with Gemini API key and concurrency settings."""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')
        self.max_workers = max_workers
        self.min_delay = 1.0 / requests_per_second  # Minimum delay between requests
        self._rate_limiter = threading.Semaphore(max_workers)
        self._last_request_time = {}
        self._lock = threading.Lock()
        
    def _rate_limit(self, worker_id: int):
        """Apply rate limiting per worker thread."""
        with self._lock:
            current_time = time.time()
            if worker_id in self._last_request_time:
                time_since_last = current_time - self._last_request_time[worker_id]
                if time_since_last < self.min_delay:
                    sleep_time = self.min_delay - time_since_last
                    time.sleep(sleep_time)
            self._last_request_time[worker_id] = time.time()
        
    def create_enhancement_prompt(self, product: Dict[str, Any]) -> str:
        """Create a detailed prompt for Gemini to enhance product data."""
        
        prompt = f"""
        You are an expert HR technology analyst. Analyze the following product/service and provide detailed enhancements in JSON format.
        
        IMPORTANT INSTRUCTIONS:
        - ONLY provide information that can be reasonably inferred from the given product data
        - DO NOT guess, assume, or make up any information
        - If there isn't enough information to populate a field accurately, either omit it entirely or provide an empty array
        - Be conservative and factual - only include data you can confidently derive from the provided information
        - If a field doesn't apply to this product type, omit it from the response
        
        PRODUCT INFORMATION:
        Name: {product.get('name', 'N/A')}
        
        {json.dumps(product, indent=2)}
        
        :PRODUCT INFORMATION END
        
        Please provide enhancements in valid JSON format. Only include fields where you have sufficient information:
        
        {{
            "enhanced_description": "Only if you can meaningfully expand on the existing description with factual information from the product data",
            "target_industries": ["Only include if clearly evident from product data - based on features, categories, or explicit mentions"],
            "ideal_company_sizes": ["Only if product data indicates specific company size targeting"],
            "key_use_cases": ["Only list use cases explicitly mentioned or clearly derivable from features/description"],
            "competitive_advantages": ["Only include advantages explicitly stated or clearly evident from features"],
            "implementation_scenarios": ["Only if product data suggests specific implementation types"],
            "hr_functions_supported": ["Only include HR functions explicitly mentioned or clearly supported based on features"],
            "technology_stack": ["Only include technologies explicitly mentioned in product data"],
            "integration_capabilities": ["Only include integrations explicitly mentioned or clearly indicated"],
            "roi_benefits": ["Only include benefits explicitly stated or directly derivable from features"],
            "searchable_keywords": ["Extract relevant keywords from existing product data - name, features, categories, tags"],
            "primary_users": ["Only include user types explicitly mentioned or clearly indicated by product features"],
            "deployment_options": ["Only if explicitly mentioned or clearly evident from product type"],
            "compliance_features": ["Only include compliance standards explicitly mentioned"],
            "scalability_features": ["Only include if scalability is explicitly mentioned or clearly evident"],
            "security_features": ["Only include security features explicitly mentioned"]
        }}
        
        Remember: It's better to provide fewer, accurate fields than to guess or make assumptions. Empty arrays or omitted fields are perfectly acceptable when there's insufficient information.
        """
        
        return prompt
    
    def enhance_product(self, product: Dict[str, Any], worker_id: int = 0) -> Dict[str, Any]:
        """Enhance a single product with additional fields using Gemini."""
        try:
            # Apply rate limiting
            self._rate_limit(worker_id)
            
            prompt = self.create_enhancement_prompt(product)
            
            logger.info(f"[Worker {worker_id}] Enhancing product: {product.get('name', 'Unknown')}")
            
            # Generate content using Gemini
            response = self.model.generate_content(prompt)
            
            # Extract JSON from response
            response_text = response.text.strip()
            
            # Clean up the response to extract JSON
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            elif "{" in response_text and "}" in response_text:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                json_text = response_text[json_start:json_end]
            else:
                json_text = response_text
            
            try:
                enhancements = json.loads(json_text)
            except json.JSONDecodeError as e:
                logger.error(f"[Worker {worker_id}] Failed to parse JSON for {product.get('name')}: {e}")
                # Provide default enhancements if parsing fails
                enhancements = self.get_default_enhancements()
            
            # Merge original product data with enhancements
            enhanced_product = product.copy()
            enhanced_product.update(enhancements)
            
            logger.info(f"[Worker {worker_id}] Successfully enhanced: {product.get('name', 'Unknown')}")
            return enhanced_product
            
        except Exception as e:
            logger.error(f"[Worker {worker_id}] Error enhancing product {product.get('name', 'Unknown')}: {e}")
            # Return original product with default enhancements
            enhanced_product = product.copy()
            enhanced_product.update(self.get_default_enhancements())
            return enhanced_product
    
    def get_default_enhancements(self) -> Dict[str, Any]:
        """Provide default enhancement structure if AI generation fails."""
        return {
            "enhanced_description": "Enhanced description not available",
            "target_industries": [],
            "ideal_company_sizes": [],
            "key_use_cases": [],
            "competitive_advantages": [],
            "implementation_scenarios": [],
            "hr_functions_supported": [],
            "technology_stack": [],
            "integration_capabilities": [],
            "roi_benefits": [],
            "searchable_keywords": [],
            "primary_users": [],
            "deployment_options": [],
            "compliance_features": [],
            "scalability_features": [],
            "security_features": []
        }
    
    def enhance_products_parallel(self, products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhance products in parallel using ThreadPoolExecutor."""
        enhanced_products = [None] * len(products)  # Maintain original order
        completed_count = 0
        total_products = len(products)
        
        logger.info(f"Starting parallel enhancement of {total_products} products with {self.max_workers} workers")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_index = {}
            for i, product in enumerate(products):
                # Create a partial function that includes the worker ID
                enhance_func = partial(self.enhance_product, worker_id=i % self.max_workers)
                future = executor.submit(enhance_func, product)
                future_to_index[future] = i
            
            # Process completed tasks
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    enhanced_product = future.result()
                    enhanced_products[index] = enhanced_product
                    completed_count += 1
                    
                    if completed_count % 10 == 0 or completed_count == total_products:
                        logger.info(f"Progress: {completed_count}/{total_products} products enhanced")
                        
                except Exception as e:
                    logger.error(f"Failed to enhance product at index {index}: {e}")
                    # Use original product if enhancement fails
                    enhanced_products[index] = products[index]
                    completed_count += 1
        
        logger.info(f"Parallel enhancement completed: {completed_count}/{total_products} products processed")
        return enhanced_products
    
    def enhance_products_batch(self, products: List[Dict[str, Any]], delay: float = 2.0) -> List[Dict[str, Any]]:
        """Legacy sequential method - kept for backward compatibility."""
        logger.warning("Using sequential processing. Consider using enhance_products_parallel() for better performance.")
        enhanced_products = []
        
        for i, product in enumerate(products):
            try:
                enhanced_product = self.enhance_product(product)
                enhanced_products.append(enhanced_product)
                
                # Rate limiting to avoid API limits
                if i < len(products) - 1:  # Don't delay after the last product
                    time.sleep(delay)
                    
            except Exception as e:
                logger.error(f"Failed to enhance product {i}: {e}")
                # Add original product if enhancement fails
                enhanced_products.append(product)
        
        return enhanced_products

def load_products(file_path: str) -> List[Dict[str, Any]]:
    """Load products from JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"File {file_path} not found")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON from {file_path}: {e}")
        return []

def save_enhanced_products(products: List[Dict[str, Any]], file_path: str):
    """Save enhanced products to JSON file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(products, f, indent=2, ensure_ascii=False)
        logger.info(f"Enhanced products saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving to {file_path}: {e}")

def load_env_file():
    """Load environment variables from .env file if it exists."""
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

def main():
    """Main function to run the product enhancement process."""
    
    # Load .env file if it exists
    load_env_file()
    
    # Configuration
    INPUT_FILE = "products.json"
    OUTPUT_FILE = "enhanced_products.json"
    MAX_WORKERS = 50  # Adjust based on API rate limits and system capacity
    REQUESTS_PER_SECOND = 4.0  # Adjust based on API rate limits
    USE_PARALLEL = True  # Set to False to use sequential processing
  
    # Load original products
    logger.info(f"Loading products from {INPUT_FILE}")
    products = load_products(INPUT_FILE)
    
    if not products:
        logger.error("No products loaded. Please check your input file.")
        return
    
    logger.info(f"Loaded {len(products)} products")
    
    # Initialize enhancer
    api_key = os.getenv("GEMINI_API_KEY", "AIzaSyCzon_smSi8Gssm3e6gXBHjEJG_HxD3yXk")
    enhancer = ProductEnhancer(api_key, max_workers=MAX_WORKERS, requests_per_second=REQUESTS_PER_SECOND)
    
    # Enhance products
    logger.info("Starting product enhancement process...")
    start_time = time.time()
    
    if USE_PARALLEL:
        enhanced_products = enhancer.enhance_products_parallel(products)
    else:
        enhanced_products = enhancer.enhance_products_batch(products, delay=1.5)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Save enhanced products
    save_enhanced_products(enhanced_products, OUTPUT_FILE)
    
    logger.info(f"Enhancement complete! {len(enhanced_products)} products enhanced and saved to {OUTPUT_FILE}")
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"ENHANCEMENT SUMMARY")
    print(f"{'='*50}")
    print(f"Input file: {INPUT_FILE}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Products processed: {len(enhanced_products)}")
    print(f"Processing mode: {'Parallel' if USE_PARALLEL else 'Sequential'}")
    print(f"Workers used: {MAX_WORKERS if USE_PARALLEL else 1}")
    print(f"Total processing time: {processing_time:.2f} seconds")
    print(f"Average time per product: {processing_time/len(enhanced_products):.2f} seconds")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()