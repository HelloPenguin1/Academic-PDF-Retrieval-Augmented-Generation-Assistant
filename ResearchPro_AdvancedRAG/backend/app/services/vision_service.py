from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from langchain.schema import Document
from langchain_core.messages import HumanMessage, SystemMessage
from config.config import llm_summarize, vision_model, groq_client, vision_instruction
import base64
import re
from unstructured.documents.elements import Image as UnstructuredImage

class MultimodalProcessor:
    def __init__(self):
        self.llm = llm_summarize
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
                category in ("Table")
                or "table" in text.lower()
            ):
                pages_with_tables.add(page)

        print(f"Detected pages with tables: {sorted(list(pages_with_tables))}")
        
    

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
                    extract_image_block_types=["Table"],
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
        
        # Regex to extract figure/table labels from surrounding text
        fig_pattern = re.compile(
            r'((?:Fig(?:ure)?|Table|Chart|Diagram|Appendix)\s*\.?\s*\d+[a-zA-Z]?(?:\s*[:\-–—]\s*[^\n]{0,120})?)',
            re.IGNORECASE
        )

        # Now process chunks using pre-computed descriptions
        for chunk in chunks:
            text = chunk.text or ""
            tables = []
            images = []

            # Extract figure/table labels from the chunk text
            labels_in_text = fig_pattern.findall(text)

            if hasattr(chunk, "metadata") and hasattr(chunk.metadata, "orig_elements"):
                for element in chunk.metadata.orig_elements:
                    if getattr(element, "category", None) == "Table":
                        html = getattr(element.metadata, "text_as_html", element.text)
                        # Try to get label from element text or caption
                        el_text = getattr(element, "text", "") or ""
                        el_labels = fig_pattern.findall(el_text)
                        tables.append({"html": html, "labels": el_labels})
                    elif isinstance(element, UnstructuredImage):
                        if hasattr(element.metadata, "image_base64"):
                            base64_img = element.metadata.image_base64
                            description = image_descriptions.get(base64_img, "[No description]")
                            # Try to get label from element caption metadata
                            caption = getattr(element.metadata, "image_caption", "") or ""
                            cap_labels = fig_pattern.findall(caption)
                            images.append({
                                "base64": base64_img,
                                "description": description,
                                "labels": cap_labels
                            })

            # Enrich page_content — embed labels so retrieval can match "Figure 2"
            enriched_content = text

            if labels_in_text:
                # Deduplicate and prepend all found labels as a searchable header
                unique_labels = list(dict.fromkeys(labels_in_text))
                enriched_content = "REFERENCES: " + " | ".join(unique_labels) + "\n\n" + enriched_content

            if tables:
                enriched_content += "\n\n--- TABLES ---\n"
                for i, tbl in enumerate(tables, 1):
                    label_str = " ".join(tbl["labels"]) if tbl["labels"] else f"Table {i}"
                    enriched_content += f"\n[{label_str}]\n{tbl['html']}\n"

            if images:
                enriched_content += "\n\n--- IMAGE DESCRIPTIONS ---\n"
                for i, img in enumerate(images, 1):
                    label_str = " ".join(img["labels"]) if img["labels"] else ""
                    prefix = f"[{label_str}] " if label_str else ""
                    enriched_content += f"\n{prefix}{img['description']}\n"

            doc = Document(
                page_content=enriched_content,
                metadata={
                    "source": "pdf",
                    "has_tables": len(tables) > 0,
                    "original_tables": [t["html"] for t in tables],
                    "has_images": len(images) > 0,
                    "original_images": [{"base64": img["base64"], "description": img["description"]} for img in images],
                    "image_description": [img["description"] for img in images],
                    "figure_labels": labels_in_text,
                    "page_number": getattr(chunk.metadata, "page_number", None),
                },
            )
            processed_docs.append(doc)

        return processed_docs

