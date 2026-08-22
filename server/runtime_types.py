"""Tiny types shared by runtime auth and the k8s launcher."""

from __future__ import annotations

from dataclasses import dataclass

from server.models import ComputerRow


@dataclass(frozen=True)
class ClusterHost:
    """Server-issued principal for Jobs that run unhosted (cloud) agents."""


Host = ComputerRow | ClusterHost
