import asyncio
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain.tools import tool

class ComparisonTool:
    """A tool for comparing products by fetching their information and creating a comparison table."""

    def __init__(self, retriever):
        """
        Initializes the ComparisonTool with a retriever.

        Args:
            retriever: An object with an async `query` method that returns a list of Document objects.
                       This will typically be the HybridSearch instance.
        """
        self.retriever = retriever

    async def _get_info_for_product(self, product_name: str) -> Dict[str, Any]:
        """Fetches and consolidates information for a single product, including image URL."""
        try:
            # Retrieve the top 3 most relevant documents for the product
            docs: List[Document] = await self.retriever.query(product_name)
            if not docs:
                return {"content": "Không tìm thấy thông tin.", "image_url": None}
            
            # Consolidate the page content of the documents
            content = "\n".join([doc.page_content for doc in docs])
            
            # Assuming image_url might be in the metadata of the first document
            image_url = docs[0].metadata.get("image_url") if docs else None
            
            return {"content": content, "image_url": image_url}
        except Exception as e:
            # logger.error(f"Error fetching info for {product_name}: {e}")
            return {"content": f"Lỗi khi truy xuất thông tin cho {product_name}.", "image_url": None}

    @tool
    async def run(self, product_names: List[str]) -> Dict[str, Any]:
        """
        Retrieves information for a list of products and formats it into a Markdown comparison table,
        along with a list of product image URLs.
        
        Args:
            product_names: A list of product names to compare.
        
        Returns:
            A dictionary with 'table' (Markdown string) and 'images' (list of dicts with 'name', 'url').
        """
        if not isinstance(product_names, list) or len(product_names) < 2:
            return {"table": "Vui lòng cung cấp ít nhất hai sản phẩm để so sánh.", "images": []}

        # Fetch information for all products concurrently
        product_results = await asyncio.gather(
            *[self._get_info_for_product(name) for name in product_names]
        )

        # Prepare data for table and images
        product_contents = [res["content"] for res in product_results]
        product_images = []
        for i, name in enumerate(product_names):
            if product_results[i]["image_url"]:
                product_images.append({"name": name, "url": product_results[i]["image_url"]})

        # Build the Markdown table
        header = "| Tính năng | " + " | ".join(product_names) + " |"
        separator = "|:--- | " + " | ".join([":---"] * len(product_names)) + " |"
        
        info_rows = [f"| **Thông tin chi tiết** | {' | '.join(content.replace('|', '\\|').replace('\n', '<br>'))} |" for content in product_contents]

        markdown_table = "\n".join([header, separator] + info_rows)

        return {"table": markdown_table, "images": product_images}

# Example of how to use it (for testing purposes)
if __name__ == '__main__':
    class MockRetriever:
        async def query(self, product_name: str) -> List[Document]:
            print(f"Mock retrieving for: {product_name}")
            if "A" in product_name:
                return [Document(page_content="Sản phẩm A: Màn hình 6.1 inch, Chip A15", metadata={"image_url": "https://example.com/phone_a.jpg"})]
            if "B" in product_name:
                return [Document(page_content="Sản phẩm B: Màn hình 6.5 inch, Chip Snapdragon 8", metadata={"image_url": "https://example.com/phone_b.jpg"})]
            return []

    async def main():
        retriever = MockRetriever()
        comp_tool = ComparisonTool(retriever)
        
        # The tool function is decorated, so we call the `run` method on the instance
        comparison_result = await comp_tool.run.ainvoke({"product_names": ["Sản phẩm A", "Sản phẩm B"]})
        
        print("--- Comparison Result ---")
        print(comparison_result)

    asyncio.run(main())
