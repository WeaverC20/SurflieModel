"""
Statistics Registry

Provides auto-registration of statistic implementations and lookup by name.
Use the @StatisticsRegistry.register decorator to register new statistics.
"""

from typing import Dict, List, Type

from .base import StatisticFunction


class StatisticsRegistry:
    """
    Central registry of all available wave statistics.

    Statistics are registered using the @register decorator on the class.
    This allows automatic discovery and enables selecting statistics by name.

    Example:
        @StatisticsRegistry.register
        class MyStatistic(StatisticFunction):
            name = "my_stat"
            ...

        # Later
        stat = StatisticsRegistry.get("my_stat")
        all_stats = StatisticsRegistry.all()
    """

    _statistics: Dict[str, StatisticFunction] = {}

    @classmethod
    def register(cls, stat_class: Type[StatisticFunction]) -> Type[StatisticFunction]:
        """
        Decorator to register a statistic class.

        Args:
            stat_class: A subclass of StatisticFunction

        Returns:
            The same class (for use as decorator)
        """
        instance = stat_class()
        cls._statistics[instance.name] = instance
        return stat_class

    @classmethod
    def get(cls, name: str) -> StatisticFunction:
        """
        Get a registered statistic by name.

        Args:
            name: The statistic's unique identifier

        Returns:
            The StatisticFunction instance

        Raises:
            KeyError: If no statistic with that name is registered
        """
        if name not in cls._statistics:
            available = ", ".join(cls._statistics.keys())
            raise KeyError(f"Unknown statistic '{name}'. Available: {available}")
        return cls._statistics[name]

    @classmethod
    def all(cls) -> List[StatisticFunction]:
        """Get all registered statistics."""
        return list(cls._statistics.values())

    @classmethod
    def names(cls) -> List[str]:
        """Get names of all registered statistics."""
        return list(cls._statistics.keys())

    @classmethod
    def clear(cls):
        """Clear all registered statistics (for testing)."""
        cls._statistics = {}
