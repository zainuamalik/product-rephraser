# Use flag --test for test mode on limited entries
# Use flag --enhance-markup for markup improvement mode

import os
import json
import logging
import pandas as pd
import argparse
import time

from typing import Dict, Any, List, Optional
from openai import OpenAI, OpenAIError
from tenacity import retry, wait_random_exponential, stop_after_attempt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from jinja2 import Template
from dotenv import load_dotenv
from logging.config import dictConfig

# ──────────────────────────────────────────────────────────────
# Inline "Config"
# ──────────────────────────────────────────────────────────────
CONFIG = {
    "model": "gpt-4o-mini",
    "temperature": 0.7,
    "max_tokens": 5000,
    "checkpoint_every": 1000,
    "chunk_size": 10000,
    "max_workers": 8,
    "test_mode_rows": 100,
    "timeout": 30,
    "batch_size": 100,
    "paths": {
        "input_csv": "data/input/products_export.csv",
        "output_csv": "data/output/rephrased_products.csv",
        "failures_csv": "data/output/rephrasing_failures.csv",
    },
}

# ──────────────────────────────────────────────────────────────
# Inline Prompts
# ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are provided with an e-commerce product listing including its title and description.\n\n"
    "Goals:\n\n"
    "1. **Rephrase the product title** so it reads more clearly and engagingly (without adding marketing buzzwords or adjectives), "
    "while preserving the brand, model and product names."
    "The rephrased title should should not exceed 200 characters.\n\n"
    "2. **Rephrase the product description** to make it clearer and more appealing. "
    "If the description contains HTML tags (e.g., <img>, <a>, etc.), preserve the HTML "
    "structure entirely but rephrase only the text within the tags to make it more engaging and descriptive.\n\n"
    "**Important Notes:**\n\n"
    "1. The rephrased content should be SEO optimized.\n"
    "2. Ensure that the rephrased content including the title and description is at least 30% different from the original content.\n"
    "3. If the description is blank, create an appropriate one using the information available from the title.\n\n"
    "**Output JSON Format:**\n"
    "```json\n"
    "{ \n"
    '  "rephrased_title": "Rephrased title here",'
    '  "rephrased_description": "Rephrased description here" \n'
    "}\n"
    "```"
)

SYSTEM_PROMPT_TITLE = (
    "You are provided with an e-commerce product listing including its title and description.\n\n"
    "Goal:\n\n"
    "**Rephrase the product title** so it reads more clearly and engagingly (without adding marketing buzzwords or adjectives), "
    "while preserving the brand, model and product names."
    "The rephrased title should should not exceed 200 characters.\n\n"
    "**Important Notes:**\n\n"
    "1. The rephrased content should be SEO optimized.\n"
    "2. Ensure that the rephrased content is at least 30% different from the original content.\n"
    "**Output JSON Format:**\n"
    '```json\n{"rephrased_title": "..."}\n```'
)

SYSTEM_PROMPT_ENHANCED = (
    "You are provided with an e-commerce product listing including its title and description.\n\n"
    "Goals:\n\n"
    "1. **Rephrase the product title** so it reads more clearly and engagingly (without adding marketing buzzwords or adjectives), "
    "while preserving the brand, model and product names. The rephrased title should not exceed 200 characters.\n\n"
    "2. **Rephrase the product description and improve the HTML markup** to make it clearer, more appealing, and better structured. "
    "Specifically:\n"
    "   - Rephrase text content for better engagement and SEO optimization\n"
    "   - Improve semantic structure using appropriate HTML tags\n"
    "   - Ensure proper nesting and accessibility\n"
    "   - Fix any broken or deprecated HTML\n"
    "   - Add semantic elements like headings, lists, and proper text formatting\n"
    "   - Optimize for both SEO and visual presentation\n\n"
    "**Important Notes:**\n\n"
    "1. Maintain all functional elements (links, images, etc.)\n"
    "2. Ensure at least 30% content difference from original.\n"
    "3. If description does not contain any markup, add appropriate HTML.\n"
    "4. If the description is blank, create an appropriate one using the information available from the title.\n\n"
    "**Output JSON Format:**\n"
    "```json\n"
    "{ \n"
    '  "rephrased_title": "...",\n'
    '  "rephrased_description": "...",\n'
    "}\n"
    "```"
)


USER_PROMPT_TEMPLATE = Template(
    """
Title: {{ title }}
Description:
\"\"\"
{{ raw_html }}
\"\"\"
""".strip()
)

# ──────────────────────────────────────────────────────────────
# Load env & setup structured logger
# ──────────────────────────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

dictConfig(
    {
        "version": 1,
        "formatters": {
            "json": {
                "format": '{"time": "%(asctime)s", "level": "%(levelname)s", "message": %(message)s }'
            }
        },
        "handlers": {
            "console": {"class": "logging.StreamHandler", "formatter": "json"},
            "file": {
                "class": "logging.FileHandler",
                "formatter": "json",
                "filename": "product_refiner.log",
            },
        },
        "root": {"handlers": ["console", "file"], "level": "INFO"},
    }
)
logger = logging.getLogger("ProductRefiner")

# ──────────────────────────────────────────────────────────────
# Function‐Calling Schemas
# ──────────────────────────────────────────────────────────────
REFINE_FN = {
    "name": "refineProductListing",
    "description": "Refine raw title + HTML description into JSON: title and description",
    "parameters": {
        "type": "object",
        "properties": {
            "rephrased_title": {"type": "string"},
            "rephrased_description": {"type": "string"},
        },
        "required": ["rephrased_title", "rephrased_description"],
    },
}

REFINE_FN_TITLE = {
    "name": "refineProductTitle",
    "description": "Refine product title only",
    "parameters": {
        "type": "object",
        "properties": {"rephrased_title": {"type": "string"}},
        "required": ["rephrased_title"],
    },
}

REFINE_FN_ENHANCED = {
    "name": "refineProductListingEnhanced",
    "description": "Refine and improve product listing with enhanced markup",
    "parameters": {
        "type": "object",
        "properties": {
            "rephrased_title": {"type": "string"},
            "rephrased_description": {"type": "string"},
        },
        "required": ["rephrased_title", "rephrased_description"],
    },
}


# ──────────────────────────────────────────────────────────────
# ProductRefiner – handles the OpenAI call + JSON parsing
# ──────────────────────────────────────────────────────────────
class ProductRefiner:
    def __init__(
        self, api_key: str, only_title: bool = False, enhance_markup: bool = False
    ):
        self.client = OpenAI(api_key=api_key, timeout=CONFIG["timeout"])
        self.model = CONFIG["model"]
        self.temp = CONFIG["temperature"]
        self.max_tokens = CONFIG["max_tokens"]
        self.only_title = only_title
        self.enhance_markup = enhance_markup

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
    def _call_fn(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            # Choose function based on mode
            if self.only_title:
                functions = [REFINE_FN_TITLE]
                function_call = {"name": REFINE_FN_TITLE["name"]}
            elif self.enhance_markup:
                functions = [REFINE_FN_ENHANCED]
                function_call = {"name": REFINE_FN_ENHANCED["name"]}
            else:
                functions = [REFINE_FN]
                function_call = {"name": REFINE_FN["name"]}

            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                functions=functions,
                function_call=function_call,
                temperature=self.temp,
                max_tokens=self.max_tokens,
            )
            fn_call = resp.choices[0].message.function_call
            raw_args = fn_call.arguments
            return json.loads(raw_args)
        except json.JSONDecodeError as e:
            logger.error(
                json.dumps(
                    {"event": "json_decode_error", "error": str(e), "raw": raw_args}
                )
            )
            raise
        except OpenAIError as e:
            if "rate_limit" in str(e).lower():
                logger.warning(json.dumps({"event": "rate_limited", "error": str(e)}))
                time.sleep(10)
            raise

    def refine(
        self, product_id: str, raw_title: str, raw_html: Optional[str]
    ) -> Dict[str, Any]:
        # Choose prompt based on mode
        if self.enhance_markup:
            system_prompt = SYSTEM_PROMPT_ENHANCED
        elif self.only_title:
            system_prompt = SYSTEM_PROMPT_TITLE
        else:
            system_prompt = SYSTEM_PROMPT

        user_prompt = USER_PROMPT_TEMPLATE.render(
            title=raw_title.strip(), raw_html=(raw_html or "")
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        result = self._call_fn(messages)

        # Handle different modes
        if self.only_title:
            return {
                "rephrased_title": result["rephrased_title"],
            }
        else:
            return {
                "rephrased_title": result["rephrased_title"],
                "rephrased_description": result["rephrased_description"],
            }


# ──────────────────────────────────────────────────────────────
# CLI: chunk+thread+checkpoint
# ──────────────────────────────────────────────────────────────
def main():
    if not OPENAI_API_KEY:
        logger.error(json.dumps({"event": "missing_api_key"}))
        return

    parser = argparse.ArgumentParser("Product Refiner")
    parser.add_argument(
        "--test",
        action="store_true",
        help=f"Only process first {CONFIG['test_mode_rows']} rows",
    )
    parser.add_argument(
        "--only-titles",
        action="store_true",
        help="Rephrase only titles (descriptions remain unchanged)",
    )
    parser.add_argument(
        "--enhance-markup",
        action="store_true",
        help="Rephrase descriptions with HTML markup improvements",
    )
    args = parser.parse_args()

    # Initialize refiner with new enhance_markup option
    refiner = ProductRefiner(
        OPENAI_API_KEY, only_title=args.only_titles, enhance_markup=args.enhance_markup
    )

    paths = CONFIG["paths"]

    # Determine output columns based on mode
    if args.enhance_markup:
        CONFIG["paths"]["output_csv"] = "data/output/rephrased_products_enhanced.csv"
        output_columns = [
            "id",
            "title",
            "description",
            "rephrased_title",
            "rephrased_description",
        ]
    elif args.only_titles:
        output_columns = ["id", "title", "description", "rephrased_title"]
    else:
        output_columns = [
            "id",
            "title",
            "description",
            "rephrased_title",
            "rephrased_description",
        ]

    # File handling and checkpointing
    processed = set()
    mode, write_header = "w", True
    output_csv_path = paths["output_csv"]

    # Check for existing output file
    if os.path.exists(output_csv_path):
        try:
            # Try to read existing file with expected columns
            done_df = pd.read_csv(output_csv_path, dtype=str, usecols=["id"])
            processed = set(done_df["id"])
            mode, write_header = "a", False
        except (KeyError, ValueError):
            # Handle case where file has different columns
            backup_path = output_csv_path + f".bak_{int(time.time())}"
            os.rename(output_csv_path, backup_path)
            logger.warning(
                json.dumps(
                    {
                        "event": "backup_old_output",
                        "backup_path": backup_path,
                        "reason": "incompatible_columns",
                    }
                )
            )

    buffer, failures = [], []
    success_count, failure_count = 0, 0

    # Configure CSV reading options
    read_opts = {"chunksize": CONFIG["chunk_size"], "dtype": str}
    if args.test:
        read_opts["nrows"] = CONFIG["test_mode_rows"]
        logger.info(
            json.dumps({"event": "test_mode", "rows": CONFIG["test_mode_rows"]})
        )

    # Process data in chunks
    for chunk in pd.read_csv(paths["input_csv"], **read_opts):
        # Filter out already processed rows
        rows = [row for _, row in chunk.iterrows() if row["id"] not in processed]

        with ThreadPoolExecutor(max_workers=CONFIG["max_workers"]) as executor:
            # Submit all rows for processing
            futures = {
                executor.submit(
                    refiner.refine, row["id"], row["title"], row.get("description")
                ): row
                for row in rows
            }

            # Process completed futures
            for fut in tqdm(
                as_completed(futures), total=len(futures), desc="Refining chunk"
            ):
                row = futures[fut]
                try:
                    result = fut.result()

                    # Build output record based on mode
                    record = {
                        "id": row["id"],
                        "title": row["title"],
                        "description": row.get("description") or "",
                    }

                    # Handle different processing modes
                    if args.enhance_markup:
                        # Enhanced markup mode - all fields
                        record.update(
                            {
                                "rephrased_title": result["rephrased_title"],
                                "rephrased_description": result[
                                    "rephrased_description"
                                ],
                            }
                        )
                    elif args.only_titles:
                        # Title-only mode
                        record["rephrased_title"] = result["rephrased_title"]
                    else:
                        # Standard mode
                        record.update(
                            {
                                "rephrased_title": result["rephrased_title"],
                                "rephrased_description": result[
                                    "rephrased_description"
                                ],
                            }
                        )

                    buffer.append(record)
                    success_count += 1

                except Exception as e:
                    # Error handling
                    failures.append({"id": row["id"], "error": str(e)})
                    failure_count += 1
                    logger.error(
                        json.dumps(
                            {"event": "refine_failed", "id": row["id"], "error": str(e)}
                        )
                    )

                # Checkpoint writing
                if len(buffer) >= CONFIG["checkpoint_every"]:
                    output_df = pd.DataFrame(buffer)[output_columns]
                    output_df.to_csv(
                        output_csv_path, mode=mode, header=write_header, index=False
                    )
                    logger.info(
                        json.dumps({"event": "checkpoint", "written": len(buffer)})
                    )
                    buffer.clear()
                    # After first write, subsequent writes should append without header
                    mode, write_header = "a", False

    # Final buffer flush
    if buffer:
        output_df = pd.DataFrame(buffer)[output_columns]
        output_df.to_csv(output_csv_path, mode=mode, header=write_header, index=False)
        logger.info(json.dumps({"event": "final_flush", "written": len(buffer)}))

    # Write failures to separate file
    if failures:
        failures_df = pd.DataFrame(failures)
        failures_df.to_csv(paths["failures_csv"], index=False)
        logger.info(json.dumps({"event": "failures_logged", "count": len(failures)}))

    # Final log
    logger.info(
        json.dumps(
            {"event": "run_complete", "success": success_count, "failed": failure_count}
        )
    )


if __name__ == "__main__":
    main()
