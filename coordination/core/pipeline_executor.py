"""
Pipeline Executor
-----------------
Orchestrates an inference request across a complete shard pipeline.

Flow:
1. Receive tokenized input from the inference API
2. Send to the first shard node (layers 0..N)
3. First node processes layers, returns activations
4. Pass activations to next node
5. Continue until final node returns output logits
6. Decode output tokens
7. Report FLOPs per node to coordination layer for token issuance
"""

import time
import requests
import json
from typing import Optional
from coordination.core.shard_registry import InferencePipeline, ShardRegistration
from coordination.core.shard_registry import registry


class PipelineExecutionError(Exception):
    pass


class PipelineExecutor:
    """
    Executes an inference request across a distributed shard pipeline.
    Handles activation passing, timeout management, and FLOP accounting.
    """

    # Timeout per shard node in seconds
    NODE_TIMEOUT = 30

    def execute(
        self,
        pipeline:      InferencePipeline,
        prompt:        str,
        max_new_tokens: int = 256,
        temperature:   float = 0.7,
    ) -> dict:
        """
        Execute a full inference request across the pipeline.

        Returns:
            {
                "text":          generated text,
                "tokens":        number of output tokens,
                "flops_per_node": {node_id: flops},
                "latency_ms":    total latency,
                "pipeline":      list of node_ids used,
            }

        Raises PipelineExecutionError if any node fails.
        """
        if not pipeline.is_complete:
            raise PipelineExecutionError("Incomplete pipeline — cannot execute.")

        start_time = time.perf_counter()

        # Step 1: Tokenize input at the first node
        first_shard = pipeline.shards[0]
        tokenize_result = self._call_node(
            shard=    first_shard,
            endpoint= "tokenize",
            payload=  {"prompt": prompt},
        )
        if not tokenize_result:
            raise PipelineExecutionError(
                f"Tokenization failed at node {first_shard.node_id}"
            )

        input_ids  = tokenize_result["input_ids"]
        input_len  = len(input_ids)

        # Step 2: Forward pass through each shard
        # Each node receives activations from previous node,
        # processes its layers, passes activations to next
        flops_per_node = {}
        current_payload = {
            "input_ids":      input_ids,
            "past_key_values": None,
            "is_first_shard":  True,
        }

        for i, shard in enumerate(pipeline.shards):
            is_last = (i == len(pipeline.shards) - 1)

            result = self._call_node(
                shard=    shard,
                endpoint= "forward",
                payload=  {
                    **current_payload,
                    "max_new_tokens": max_new_tokens if is_last else 0,
                    "temperature":    temperature,
                    "is_last_shard":  is_last,
                },
            )

            if not result:
                registry.mark_offline(shard.node_id)
                raise PipelineExecutionError(
                    f"Forward pass failed at node {shard.node_id} "
                    f"(layers {shard.layer_start}-{shard.layer_end})"
                )

            # Accumulate FLOPs for this node
            flops_per_node[shard.node_id] = result.get("flops_delivered", 0.0)

            # Record in registry
            registry.record_request(
                shard.node_id,
                shard.model_id,
                flops_per_node[shard.node_id]
            )

            if is_last:
                # Final node returns generated tokens
                output_ids = result.get("output_ids", [])
                output_text = result.get("text", "")
            else:
                # Pass activations to next node
                current_payload = {
                    "activations":    result.get("activations"),
                    "past_key_values": result.get("past_key_values"),
                    "input_ids":      input_ids,
                    "is_first_shard": False,
                }

        latency_ms = (time.perf_counter() - start_time) * 1000

        return {
            "text":           output_text,
            "tokens":         len(output_ids) if output_ids else 0,
            "input_tokens":   input_len,
            "flops_per_node": flops_per_node,
            "total_flops":    sum(flops_per_node.values()),
            "latency_ms":     round(latency_ms, 2),
            "pipeline":       pipeline.node_ids,
        }

    def _call_node(
        self,
        shard:    ShardRegistration,
        endpoint: str,
        payload:  dict,
    ) -> Optional[dict]:
        """
        Call an activation server endpoint on a shard node.
        Returns parsed response or None on failure.
        """
        url = f"http://{shard.host}:{shard.port}/shard/{endpoint}"
        try:
            resp = requests.post(
                url,
                json=    payload,
                timeout= self.NODE_TIMEOUT,
            )
            if resp.status_code == 200:
                return resp.json()
            return None
        except requests.exceptions.Timeout:
            return None
        except requests.exceptions.ConnectionError:
            return None
        except Exception:
            return None


# Singleton instance
pipeline_executor = PipelineExecutor()
