"""
CAN Bus Integration Module

This module provides CAN bus communication capabilities for vehicle telemetry
and control commands using SocketCAN on Linux.
"""

# Auto-generated exports
from .command_sender import CommandSender
from .dbc_parser import DBCParser, Message, Signal
from .interface import CANInterface
from .logger import CANLogger, CANPlayback
from .telemetry_reader import TelemetryReader

__all__ = [
    'CANInterface',
    'CANLogger',
    'CANPlayback',
    'CommandSender',
    'DBCParser',
    'Message',
    'Signal',
    'TelemetryReader',
]
