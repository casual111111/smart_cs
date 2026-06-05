__all__ = [
    "ConversationMemory",
    "MemoryMessage",
]


def __getattr__(name: str):
    if name in __all__:
        from app.memory.short_term_memory import ConversationMemory, MemoryMessage

        exports = {
            "ConversationMemory": ConversationMemory,
            "MemoryMessage": MemoryMessage,
        }
        return exports[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
