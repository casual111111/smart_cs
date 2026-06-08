class SimpleTokenCounter:
    """Lightweight token estimate without tokenizer dependencies."""

    def count(self, text: str) -> int:
        if not text:
            return 0

        ascii_chars = 0
        non_ascii_chars = 0

        for char in text:
            if char.isspace():
                continue
            if ord(char) < 128:
                ascii_chars += 1
            else:
                non_ascii_chars += 1

        return max(1, (ascii_chars + 3) // 4 + non_ascii_chars)

    def truncate(self, text: str, max_tokens: int) -> str:
        if self.count(text) <= max_tokens:
            return text

        if max_tokens <= 0:
            return ""

        result = []
        used = 0

        for char in text:
            if char.isspace():
                cost = 0
            elif ord(char) < 128:
                cost = 0.25
            else:
                cost = 1

            if used + cost > max_tokens:
                break

            result.append(char)
            used += cost

        return "".join(result).rstrip()
