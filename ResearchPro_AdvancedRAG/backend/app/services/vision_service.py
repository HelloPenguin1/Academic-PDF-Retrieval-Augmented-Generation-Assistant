from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from langchain.schema import Document
from langchain_core.messages import HumanMessage, SystemMessage
from config.config import llm_summarize, vision_model, groq_client, vision_instruction
import base64
from unstructured.documents.elements import Image as UnstructuredImage

class MultimodalProcessor:
    def __init__(self):
        self.llm = llm_summarize
        self.vision_model = vision_model
        self.groq_client = groq_client
        self.image_cache = {}
        

    def load_and_process(self, filepath: str) -> list[Document]:
        print("Fast scan to detect table/image pages...")
        fast_scan = partition_pdf(
            filename=filepath,  strategy="fast", infer_table_structure=False, extract_image_block_types=None, languages=["eng"]
        )
        pages_with_tables = set()

        # Detect which pages need hi_res
        for el in fast_scan:
            category = getattr(el, "category", None)
            text = getattr(el, "text", "") or ""
            page = getattr(el.metadata, "page_number", None)

            if page is None:
                continue
            if (
                category in ("Table", "Image", "Graphic", "Figure")
                or "table" in text.lower()
                or "figure" in text.lower()
                or "chart" in text.lower()
                or "diagram" in text.lower()
                or "plot" in text.lower()
            ):
                pages_with_tables.add(page)

        print(f"Detected table/image pages: {sorted(list(pages_with_tables))}")
        
    

        # Step 2: If no tables/images → use fast output only
        if not pages_with_tables:
            print("No complex elements detected. Using fast scan for all pages.")
            elements = fast_scan
        else:
            print("Running hi_res selectively on visual pages...")
            hi_res_elements = partition_pdf(
                    filename=filepath,
                    strategy="hi_res",
                    infer_table_structure=True,
                    extract_image_block_types=["Table", "Image", "Figure"],
                    extract_image_block_to_payload=True,
                    languages=["eng"],
                    page_range=",".join(str(p) for p in pages_with_tables)
                )
           

            # Merge the elements 
            elements = []
            for el in fast_scan:
                page = getattr(el.metadata, "page_number", None)
                if page in pages_with_tables:
                    continue
                elements.append(el)

            elements.extend(list(hi_res_elements))
            elements.sort(key=lambda el: getattr(el.metadata, "page_number", 0))

        # Step 3: Chunking
        print("Chunking by title...")
        chunks = chunk_by_title(
            elements,
            max_characters=1500,
            new_after_n_chars=1200,
            combine_text_under_n_chars=300
        )

        # Step 4: Convert to Document objects 
        processed_docs = self._convert_chunks_without_summary(chunks)

        return processed_docs
    
    

    def describe_image(self, base64_img: str) -> str:
        
        if len(base64_img) > 2_000_000:
            return "[Image too large to analyze]"
        
        if base64_img in self.image_cache:
            return self.image_cache[base64_img]
        
        try:
            # Clean base64 (same as test.py implicitly handles)
            base64_img = base64_img.replace("\n", "").replace(" ", "")
            base64_img = base64_img + "=" * (-len(base64_img) % 4)
            if base64_img.startswith("iVBOR"):
                mime = "image/png"
            else:
                mime = "image/jpeg"

            image_url = f"data:{mime};base64,{base64_img}"

            response = self.groq_client.chat.completions.create(
                model=vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": vision_instruction},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url
                                },
                            },
                        ],
                    }
                ],
                max_tokens=300,
                temperature=0,
            )

            desc = response.choices[0].message.content[:500]
            return desc
        
        except Exception as e:
            return "[Image could not be analyzed]"


    def _convert_chunks_without_summary(self, chunks) -> list[Document]:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        processed_docs = []
        
        # Collect all images first
        all_images_to_describe = []
        for chunk in chunks:
            if hasattr(chunk, "metadata") and hasattr(chunk.metadata, "orig_elements"):
                for element in chunk.metadata.orig_elements:
                    if isinstance(element, UnstructuredImage):
                        if hasattr(element.metadata, "image_base64"):
                            all_images_to_describe.append(element.metadata.image_base64)
        
        # Describe all images in parallel (max 4 concurrent)
        image_descriptions = {}
        if all_images_to_describe:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(self.describe_image, img): img 
                    for img in all_images_to_describe
                }
                for future in as_completed(futures):
                    img = futures[future]
                    try:
                        image_descriptions[img] = future.result(timeout=30)
                    except Exception:
                        image_descriptions[img] = "[Image analysis failed]"
        
        # Now process chunks using pre-computed descriptions
        for chunk in chunks:
            text = chunk.text or ""
            tables = []
            images = []

            if hasattr(chunk, "metadata") and hasattr(chunk.metadata, "orig_elements"):
                for element in chunk.metadata.orig_elements:
                    if getattr(element, "category", None) == "Table":
                        html = getattr(element.metadata, "text_as_html", element.text)
                        tables.append(html)
                    elif isinstance(element, UnstructuredImage):
                        if hasattr(element.metadata, "image_base64"):
                            base64_img = element.metadata.image_base64
                            # Use pre-computed description
                            description = image_descriptions.get(base64_img, "[No description]")
                            images.append({
                                "base64": base64_img,
                                "description": description
                            })

            # Enrich page_content with tables and image descriptions
            enriched_content = text
            
            if tables:
                enriched_content += "\n\n--- TABLES ---\n"
                for i, table_html in enumerate(tables, 1):
                    enriched_content += f"\n[TABLE {i}]\n{table_html}\n"
            
            if images:
                enriched_content += "\n\n--- IMAGE DESCRIPTIONS ---\n"
                for i, img in enumerate(images, 1):
                    enriched_content += f"\n[IMAGE {i}]: {img['description']}\n"

            doc = Document(
                page_content=enriched_content,
                metadata={
                    "source": "pdf",
                    "has_tables": len(tables) > 0,
                    "original_tables": tables,
                    "has_images": len(images) > 0,
                    "original_images": images,
                    "image_description": [img["description"] for img in images],
                    "page_number": getattr(chunk.metadata, "page_number", None),
                },
            )
            processed_docs.append(doc)

        return processed_docs

    

    # def _generate_ai_summary(self, text, tables, images) -> str:        
    #     # Construct context for the LLM
    #     context_str = f"TEXT:\n{text}\n\n"
    #     for i, table in enumerate(tables[:3]): #max 3 tables
    #         context_str += f"TABLE {i+1}:\n{table}\n\n"
    
    #     if images:
    #         context_str += f"\n[{len(images)} IMAGE(S) WITH DESCRIPTIONS]\n"
    #         for i, img in enumerate(images[:3], 1):
    #             description = img.get("description", "No description available")
    #             if len(description) > 400:
    #                 description = description[:400] + "..." 
    #             context_str += f"\nImage {i}: {description}\n"

    #     prompt = f"""You are a concise research summarizer. Create a brief, searchable summary integrating text, tables, and visual elements.

    #     SUMMARY RULES:
    #     - Extract core findings, methodology, and key results from text
    #     - Convert tables to brief data statements with exact numbers (e.g., "Accuracy increased from 50% to 85%")
    #     - Synthesize key insights from image descriptions—extract metrics, trends, and patterns
    #     - Preserve domain terms, metric names, and visual insights for searchability
    #     - Never invent data—only use what's explicitly shown
    #     - Keep under 200 words; prioritize key insights over completeness

    #     CONTENT:
    #     {context_str}

    #     SUMMARY (direct, no formatting tags):"""

    #     try:
    #         response = self.llm.invoke([HumanMessage(content=prompt)])
    #         return response.content
    #     except Exception as e:
    #         print(f"Summary failed: {e}")
    #         return text
