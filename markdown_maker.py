
"""
Universal OCR Markdown Generator
Works with ANY document type - bills, passbooks, assessments, forms, receipts, etc.
Zero hardcoded keywords or patterns - purely spatial layout analysis
"""

import json
from typing import List, Dict, Any, Tuple
from collections import defaultdict

class UniversalMarkdownGenerator:
    """
    Truly universal markdown generator using:
    - Spatial clustering (automatic region detection)
    - Dynamic pattern learning (learns document structure)
    - Column detection (multi-column layout support)
    - Smart line grouping (proximity-based merging)

    NO hardcoded keywords, NO document-specific logic
    """

    def __init__(self, 
                 min_confidence: float = 0.20,
                 y_line_tolerance: float = 0.015,
                 x_adjacency_gap: float = 0.05,
                 column_gap_threshold: float = 0.15):
        """
        Initialize with configurable thresholds

        Args:
            min_confidence: Minimum OCR confidence (0.0-1.0)
            y_line_tolerance: Y-axis tolerance for same-line grouping
            x_adjacency_gap: Maximum X-gap for adjacent item merging
            column_gap_threshold: X-gap that indicates new column
        """
        self.min_confidence = min_confidence
        self.y_line_tolerance = y_line_tolerance
        self.x_adjacency_gap = x_adjacency_gap
        self.column_gap_threshold = column_gap_threshold

    def generate_markdown(self, ocr_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate markdown from OCR results

        Args:
            ocr_data: Dictionary with filename as key and OCR content

        Returns:
            Dictionary with markdown and raw_ocr for each file
        """
        result = {}

        for filename, data in ocr_data.items():
            try:
                content = data.get('content', data) if isinstance(data, dict) else data
                markdown = self._generate_universal_markdown(content, filename)

                result[filename] = {
                    "markdown": markdown,
                    "raw_ocr": content
                }
            except Exception as e:
                result[filename] = {
                    "markdown": f"# {filename}\n\n*Error: {str(e)}*",
                    "raw_ocr": []
                }

        return result

    def _generate_universal_markdown(self, content: List, filename: str) -> str:
        """Generate markdown using spatial analysis only"""

        if not content:
            return f"# {filename}\n\n*No content found*"

        # Step 1: Parse and clean items
        items = self._parse_items(content)

        if not items:
            return f"# {filename}\n\n*No valid content found*"

        # Step 2: Detect columns dynamically
        columns = self._detect_columns(items)

        # Step 3: Process each column separately
        column_lines = []
        for col_items in columns:
            lines = self._group_into_lines(col_items)
            column_lines.append(lines)

        # Step 4: Merge columns intelligently (by Y-coordinate)
        merged_lines = self._merge_columns(column_lines)

        # Step 5: Detect structure patterns (lists, key-values, headings)
        structured_content = self._detect_structure(merged_lines)

        # Step 6: Build markdown
        return self._build_markdown(structured_content, filename)

    def _parse_items(self, content: List) -> List[Dict]:
        """Parse and filter OCR items"""
        items = []

        for item in content:
            try:
                if len(item) >= 3:
                    text = str(item[0]).strip()
                    confidence = float(item[1])
                    coords = item[2] if len(item[2]) == 4 else [0, 0, 0, 0]

                    # Universal noise filtering (symbols only)
                    if (confidence >= self.min_confidence and 
                        text and 
                        len(text) > 0 and
                        text not in ['*', '**', '***', '•', '·', '─', '│']):

                        items.append({
                            'text': text,
                            'x': float(coords[0]),
                            'y': float(coords[1]),
                            'width': float(coords[2]),
                            'height': float(coords[3]),
                            'confidence': confidence
                        })
            except:
                continue

        return items

    def _detect_columns(self, items: List[Dict]) -> List[List[Dict]]:
        """
        Dynamically detect columns based on X-coordinate gaps
        No hardcoded column positions
        """
        if not items:
            return []

        # Sort by X coordinate
        sorted_items = sorted(items, key=lambda x: x['x'])

        columns = []
        current_column = [sorted_items[0]]
        last_x_end = sorted_items[0]['x'] + sorted_items[0]['width']

        for item in sorted_items[1:]:
            x_gap = item['x'] - last_x_end

            # If large gap, start new column
            if x_gap > self.column_gap_threshold:
                columns.append(current_column)
                current_column = [item]
            else:
                current_column.append(item)

            last_x_end = max(last_x_end, item['x'] + item['width'])

        if current_column:
            columns.append(current_column)

        return columns

    def _group_into_lines(self, items: List[Dict]) -> List[Dict]:
        """
        Group items into lines based on Y-coordinate proximity
        """
        if not items:
            return []

        # Sort by Y then X
        sorted_items = sorted(items, key=lambda x: (x['y'], x['x']))

        lines = []
        current_line_items = [sorted_items[0]]
        current_y = sorted_items[0]['y']

        for item in sorted_items[1:]:
            y_diff = abs(item['y'] - current_y)

            if y_diff <= self.y_line_tolerance:
                # Check X-proximity for adjacent merging
                last_item = current_line_items[-1]
                x_gap = item['x'] - (last_item['x'] + last_item['width'])

                if x_gap <= self.x_adjacency_gap:
                    current_line_items.append(item)
                else:
                    # Same Y but not adjacent - create separate line
                    lines.append(self._create_line(current_line_items))
                    current_line_items = [item]
                    current_y = item['y']
            else:
                # Different Y - new line
                lines.append(self._create_line(current_line_items))
                current_line_items = [item]
                current_y = item['y']

        if current_line_items:
            lines.append(self._create_line(current_line_items))

        return lines

    def _create_line(self, items: List[Dict]) -> Dict:
        """Create a line object from items"""
        items.sort(key=lambda x: x['x'])
        text = ' '.join([item['text'] for item in items])

        return {
            'text': text,
            'y': items[0]['y'],
            'x': items[0]['x'],
            'items': items,
            'avg_height': sum(item['height'] for item in items) / len(items)
        }

    def _merge_columns(self, column_lines: List[List[Dict]]) -> List[Dict]:
        """
        Merge lines from different columns based on Y-coordinate
        Keeps columns separate when appropriate
        """
        if not column_lines:
            return []

        # Flatten all lines
        all_lines = []
        for lines in column_lines:
            all_lines.extend(lines)

        # Sort by Y-coordinate
        all_lines.sort(key=lambda x: x['y'], reverse=True)

        return all_lines

    def _detect_structure(self, lines: List[Dict]) -> List[Dict]:
        """
        Detect structure patterns: lists, key-values, headings, paragraphs
        Uses only formatting cues (no keywords)
        """
        structured = []

        for line in lines:
            text = line['text']

            # Detect type based on formatting only
            line_type = self._classify_line(text, line)

            structured.append({
                'text': text,
                'type': line_type,
                'y': line['y']
            })

        return structured

    def _classify_line(self, text: str, line: Dict) -> str:
        """
        Classify line type based on formatting patterns only
        """
        # Heading: ALL CAPS, short, or larger height
        if (text.isupper() and len(text.split()) <= 5) or line['avg_height'] > 0.025:
            return 'heading'

        # Key-value: Contains colon or equals
        elif ':' in text or '=' in text:
            return 'key_value'

        # List item: Starts with dash, hyphen, or contains ' - '
        elif text.startswith('-') or text.startswith('•') or ' - ' in text or ' = ' in text:
            return 'list_item'

        # Default: paragraph
        else:
            return 'paragraph'

    def _build_markdown(self, structured: List[Dict], filename: str) -> str:
        """Build markdown from structured content"""

        md = [f"# {filename}\n"]

        current_section = None
        in_list = False

        for item in structured:
            text = item['text']
            line_type = item['type']

            # Handle headings
            if line_type == 'heading':
                if in_list:
                    md.append("")  # Close list
                    in_list = False
                md.append(f"## {text}\n")
                current_section = text

            # Handle key-value pairs
            elif line_type == 'key_value':
                if in_list:
                    md.append("")  # Close list
                    in_list = False

                formatted = self._format_key_value(text)
                if formatted:
                    md.append(formatted)

            # Handle list items
            elif line_type == 'list_item':
                if not in_list:
                    in_list = True
                md.append(f"- {text}")

            # Handle paragraphs
            else:
                if in_list:
                    md.append("")  # Close list
                    in_list = False
                md.append(f"{text}  ")

        # Close final list if needed
        if in_list:
            md.append("")

        return "\n".join(md)

    def _format_key_value(self, text: str) -> str:
        """Format key-value pairs"""

        # Try colon separator
        if ':' in text:
            parts = text.split(':', 1)
            key = parts[0].strip()
            value = parts[1].strip().lstrip(':').strip() if len(parts) > 1 else ""

            if value:
                return f"**{key}:** {value}  "

        # Try equals separator
        elif '=' in text:
            parts = text.split('=', 1)
            key = parts[0].strip()
            value = parts[1].strip() if len(parts) > 1 else ""

            if value:
                return f"**{key}:** {value}  "

        return ""

# Convenience function
def generate_markdown_from_json(json_data: str) -> str:
    """Generate markdown from JSON string"""
    generator = UniversalOCRMarkdownGenerator()
    data = json.loads(json_data)
    result = generator.generate_markdown(data)
    return json.dumps(result, indent=2, ensure_ascii=False)

# Example usage
if __name__ == "__main__":
    # Can be configured for different document types

    # For dense documents (bills, forms)
    dense_generator = UniversalOCRMarkdownGenerator(
        y_line_tolerance=0.01,
        x_adjacency_gap=0.03,
        column_gap_threshold=0.12
    )

    # For sparse documents (letters, reports)
    sparse_generator = UniversalOCRMarkdownGenerator(
        y_line_tolerance=0.02,
        x_adjacency_gap=0.08,
        column_gap_threshold=0.20
    )

    print("Universal OCR Markdown Generator - Ready")
    print("Configurable for any document type")