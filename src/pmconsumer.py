"""
Implements the start code for pmhistory consumer.
"""

# !!!!! We must set this environment variable before importing asyncio !!!!!
import os
from dataset import Dataset
import tempfile
from typing import Dict
from constants import COUNTER_NAMES, SERVICE_NAME

os.environ["PROMETHEUS_MULTIPROC_DIR"] = tempfile.mkdtemp(prefix=str(os.getpid()))

import random  # noqa: E402
from typing import List  # noqa: E402
from prometheus_client import Counter, Gauge  # noqa: E402
from rapplib import (
    rapp,
    rapp_base,
    data_processor,
    metrics,
)  # noqa: E402

# Constants for the rapp
MAX_FREQ_BAND = 102

CM_PARAMETER_NAME = "freqBand"
CMREAD_DIRECT = "cmread_direct"



class PmHistoryConsumerProcessor(data_processor.DataProcessor):
    """
    Overrides data_processor_class
    """

    def __init__(self, stop_event, log, config, dataset):
        # Keep track of the cell ids seen in the data process. We are single
        # threaded there so safe to access.
        self.processed_cell_ids = set()
        # These are the cells seen in the current loop only.
        self.unprocessed_cell_ids = set()

        super().__init__(stop_event, log, config)
        self.dataset : Dataset = dataset

    def pmhistory_data_handler(self, job_id, data_type, data):
        """
        Make a prometheus update and keep track of cells.
        """
        if data:
            cell_id = data["cell_global_id"]
            labels = dict(self.config.prometheus_gauge_labels)
            labels.update(
                {
                    "name": data["counter_name"],
                    "cell": cell_id,
                }
            )

            self.dataset.augment_data(data)

            self.config.counters_cells_values.labels(
                **labels,
            ).set(float(data["counter_value"]))

            if cell_id not in self.processed_cell_ids:
                self.unprocessed_cell_ids.add(cell_id)

        # Batch the CM Handling for every 500 cells, but only from
        # pmhistory handling.
        if len(self.unprocessed_cell_ids) >= 500:
            self.perform_cm_handling()

    def job_processed_handler(self, job_id, data_type):
        """
        Deal with a job indicating it is complete. First do the common actions
        on the super, then handling any seen cell ids which have not been
        processed yet.
        """
        super().job_processed_handler(job_id, data_type)
        # Process the cell ids which have not yet been processed from completed
        # pmhistory jobs and clear cell id tracker.
        if data_type == rapp.pm_history_info_type:
            if len(self.unprocessed_cell_ids) > 0:
                self.perform_cm_handling()
                self.log.info(
                    "saw %d cells from %s job %s",
                    len(self.processed_cell_ids),
                    rapp.pm_history_info_type,
                    job_id,
                )
            self.processed_cell_ids.clear()

    def perform_cm_handling(self):
        """
        Function to perform CM handling by writing a CM parameter for all
        the cells received in the PM history data and any cells that failed
        to be udpated during the cmwrite operation. The parameter if freqBand.
        It is set to a random value.

        This used to do a cmread_direct for each cell before doing the mass
        write, but this was thousands of cells, causing the queue to block
        such that job results were not getting processed in a timely fashion.
        """

        self.log.info("Notifying processing for unprocessed cell_ids")

        cm_data_value_list: List[dict] = []
        for cell_id in self.unprocessed_cell_ids:
            parameter_value = random.randint(1, MAX_FREQ_BAND)
            cm_data_value = {
                "cell_id": cell_id,
                "parameter_name": CM_PARAMETER_NAME,
                "parameter_value": parameter_value,
            }
            cm_data_value_list.append(cm_data_value)

        if not cm_data_value_list:
            raise rapp.RappError("cm data list is required to register a job")

        # request cmwrite job
        cm_data = {
            "values": cm_data_value_list,
        }

        self.log.info("sending cmwrite job with %d values", len(cm_data_value_list))

        job_data = {
            "data_type": rapp.cmwrite_info_type,
            "job_definition": cm_data,
        }
        self.job_request(job_data)

        # request cmread_direct
        # note that we cannot guarantee that the cmreads will happen after
        # the cmwrites. It's used to demonstrate different types of job
        # related requests.
        for cell_id in self.unprocessed_cell_ids:
            cmread_data = {"cell_id": cell_id, "parameter_name": CM_PARAMETER_NAME}
            self.send_cmread_direct_request(cmread_data)

        self.unprocessed_cell_ids.clear()

    def send_cmread_direct_request(self, cmread_data: Dict[str, str]):
        """
        Puts a cmread_direct event type to management queue, which is handled
        by the main process.
        @param cmread_data is a dictionary of "cell_id" and "parameter_name".
        """
        self.send_management_event(CMREAD_DIRECT, cmread_data)

    def cmread_direct_data_handler(self, job_id, data_type, cmread_data):
        """
        This should be overriden by something useful
        """
        self.log.info("Received updated information: %s", cmread_data)
        # the updated CM read data can be used for further processing

    def cmwrite_data_handler(self, job_id, data_type, data):
        """
        Acknowledge cmwrite response. For any failed writes, write those
        cell ids to the log, but do not try again as we'll conflict with
        the handling that the pmhistory job is doing.
        """
        self.log.info(
            "for job %s received cmwrite job results of len %d", job_id, len(data)
        )
        failed_rows = set()
        for result_row in data:
            status = result_row["status"]
            if result_row["status"] != "SUCCESS":
                cell = result_row["result"]["cell_id"]
                parameter = result_row["result"]["parameter_name"]
                failed_rows.add((cell, parameter, status))
        if failed_rows:
            self.log.info("for job %s some cm writes failed: %s", job_id, failed_rows)


class PmHistoryConsumer(rapp_base.RAppBase):
    """
    Class for the pmhistory consumer rapp
    """

    service_name = SERVICE_NAME
    service_port = "9080"
    service_prefix = "/v0alpha1"
    service_version = "0.1.0"
    service_display_name = "PM Historical Python based Data Consumer"
    service_description = "Python rApp that consumes historical PM counter data"

    pmhistory_counter_names = COUNTER_NAMES

    # Overridding data_processor
    data_processor_class = PmHistoryConsumerProcessor

    counters_cells_values = Gauge(
        name="counters_cells_values",
        documentation="Value of the named pmcounter in the current 15 "
        "minutes window, by cell.",
        labelnames=metrics.DEFAULT_METRIC_LABELS + ["name", "cell"],
        # We only expect one value at a time on this gauge, but we need to
        # choose a mode in case more than one process does provide a value.
        multiprocess_mode="max",
    )