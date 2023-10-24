"""Collect and combine all the slow slip catalogs for the prpose of the global analysis. 

This module is kept separate as its not explicitely part of the data preparation and rather reflects 
potential decisions in analysing the global catalog.

Typical usage example:

slowslip = AllSlowSlip(depth_subset='shallow')
"""
from typing import Literal, Optional
import numpy as np
from src.catalogs import (
    JapanSlowSlipCatalog,
    MexicoSlowSlipCatalog,
    XieSlowSlipCatalog,
    WilliamsSlowSlipCatalog,
    MichelSlowSlipCatalog,
    OkadaAlaskaSlowSlipCatalog,
)


class AllSlowSlip:
    def __init__(
        self,
        depth_subset: Literal["shallow", "deep", "all"] = "all",
        duration_subset: Literal["short term", "long term", "all"] = "all",
        mag_min: Optional[float] = None,
        impute_duration: bool = True,
        duplicate_radius: float = 70,
        duplicate_time: float = 40,
    ) -> list:
        # Consider depth subsets THIS CURRENTLY RELOADS THE DATA!!
        if depth_subset == "shallow":
            shallow_slowslip = [
                JapanSlowSlipCatalog().get_ryukyu_trench().get_clusters("depth", 2)[0],
                JapanSlowSlipCatalog().get_japan_trench().get_clusters("depth", 2)[0],
                XieSlowSlipCatalog().get_clusters("depth", 2)[0],
                WilliamsSlowSlipCatalog().get_clusters("depth", 2)[0],
                OkadaAlaskaSlowSlipCatalog.get_clusters("depth", 2)[0],
            ]

            slowslip = shallow_slowslip

        elif depth_subset == "deep":
            deep_slowslip = [
                JapanSlowSlipCatalog().get_nankai_trough().get_clusters("depth", 2)[1],
                JapanSlowSlipCatalog().get_ryukyu_trench().get_clusters("depth", 2)[1],
                JapanSlowSlipCatalog().get_japan_trench().get_clusters("depth", 2)[1],
                MexicoSlowSlipCatalog(),
                XieSlowSlipCatalog().get_clusters("depth", 2)[1],
                WilliamsSlowSlipCatalog().get_clusters("depth", 2)[1],
                MichelSlowSlipCatalog(),
                OkadaAlaskaSlowSlipCatalog.get_clusters("depth", 2)[1],
            ]

            slowslip = deep_slowslip

        elif depth_subset == "all":
            slowslip = [
                JapanSlowSlipCatalog().get_nankai_trough(),
                JapanSlowSlipCatalog().get_ryukyu_trench(),
                JapanSlowSlipCatalog().get_japan_trench(),
                MexicoSlowSlipCatalog(),
                XieSlowSlipCatalog(),
                WilliamsSlowSlipCatalog(),
                MichelSlowSlipCatalog(),
                OkadaAlaskaSlowSlipCatalog(),
            ]

        # Consider magnitude cutoff
        for ss in slowslip:
            ss.mag_completeness = mag_min  # changing in place

        # Consider different duration types of SSEs
        if duration_subset == "short term":
            slowslip = [ss.get_short_term_events() for ss in slowslip]
        elif duration_subset == "long term":
            slowslip = [ss.get_long_term_events() for ss in slowslip]

        slowslip = [ss for ss in slowslip if len(ss) > 0]
        all_slowslip = sum(slowslip[1:], slowslip[0])

        # Consider unique events
        filter_kwargs = dict(
            buffer_radius_km=duplicate_radius,
            buffer_time_days=duplicate_time,
            stategy="reference",
        )

        all_slowslip = all_slowslip.filter_duplicates(
            **filter_kwargs,
            ref_preference=all_slowslip.catalog.groupby("ref")
            .max()
            .time.sort_values(ascending=False)
            .index
        )

        slowslip_filtered = []
        for i_slowslip in slowslip:
            slowslip_filtered.append(
                i_slowslip.filter_duplicates(
                    **filter_kwargs,
                    ref_preference=i_slowslip.catalog.groupby("ref")
                    .max()
                    .time.sort_values(ascending=False)
                    .index
                )
            )

        slowslip = slowslip_filtered
        slowslip = sorted(slowslip, key=lambda x: len(x), reverse=True)
        for i_slowslip in slowslip:
            i_slowslip.catalog["name"] = i_slowslip.name
            i_slowslip.catalog["region"] = i_slowslip.region
        all_slowslip = sum(slowslip[1:], slowslip[0])

        if impute_duration is True:
            median_duration = np.nanmedian(all_slowslip.catalog.duration)
            all_slowslip.catalog.duration.fillna(median_duration)
            for i_slowslip in slowslip:
                i_slowslip.catalog.duration.fillna(median_duration, inplace=True)