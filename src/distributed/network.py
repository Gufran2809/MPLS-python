"""Network communication module for distributed MPLS.

This module provides gRPC-based communication between workers including:
- Layer exchange (pulling specific layers from peers)
- Metadata sharing (data distribution, status)
- Bandwidth monitoring
- Connection management
"""

import io
import time
import logging
from concurrent import futures
from typing import Dict, List, Optional, Tuple

import grpc
import torch
import torch.nn as nn
import numpy as np

from . import mpls_service_pb2
from . import mpls_service_pb2_grpc


logger = logging.getLogger(__name__)


class TensorSerializer:
    """Handles serialization and deserialization of PyTorch tensors."""
    
    @staticmethod
    def serialize_tensor(tensor: torch.Tensor) -> Tuple[bytes, List[int], str]:
        """Serialize a tensor to bytes.
        
        Args:
            tensor: PyTorch tensor to serialize
            
        Returns:
            Tuple of (bytes, shape, dtype_string)
        """
        buffer = io.BytesIO()
        # Convert to CPU and numpy for serialization
        np_array = tensor.detach().cpu().numpy()
        np.save(buffer, np_array, allow_pickle=False)
        return buffer.getvalue(), list(tensor.shape), str(tensor.dtype)
    
    @staticmethod
    def deserialize_tensor(data: bytes, shape: List[int], dtype: str, device: str = 'cpu') -> torch.Tensor:
        """Deserialize bytes to a tensor.
        
        Args:
            data: Serialized tensor data
            shape: Original tensor shape
            dtype: Original tensor dtype
            device: Device to place tensor on
            
        Returns:
            Reconstructed PyTorch tensor
        """
        buffer = io.BytesIO(data)
        np_array = np.load(buffer, allow_pickle=False)
        
        # Convert dtype string to torch dtype
        dtype_map = {
            'torch.float32': torch.float32,
            'torch.float16': torch.float16,
            'torch.int64': torch.int64,
        }
        torch_dtype = dtype_map.get(dtype, torch.float32)
        
        tensor = torch.from_numpy(np_array).to(dtype=torch_dtype, device=device)
        return tensor


class MPLSServicer(mpls_service_pb2_grpc.MPLSServiceServicer):
    """gRPC servicer implementation for MPLS operations."""
    
    def __init__(self, worker):
        """Initialize servicer with reference to worker.
        
        Args:
            worker: DistributedWorker instance
        """
        self.worker = worker
        self.serializer = TensorSerializer()
    
    def PullLayers(self, request, context):
        """Handle layer pull requests from peers.
        
        Args:
            request: LayerRequest message
            context: gRPC context
            
        Returns:
            LayerResponse with requested layers
        """
        try:
            logger.debug(f"Worker {self.worker.config.worker_id} received layer request "
                        f"from worker {request.requester_id} for layers {list(request.layer_indices)}")
            
            # Get current model
            model = self.worker.get_current_model()
            layers = self.worker.extract_layers(model)
            
            # Serialize requested layers
            layer_data_list = []
            for layer_idx in request.layer_indices:
                if layer_idx < len(layers):
                    layer = layers[layer_idx]
                    # Get layer parameters
                    params = self.worker.get_layer_params(layer)
                    
                    # Serialize
                    data, shape, dtype = self.serializer.serialize_tensor(params)
                    
                    layer_data = mpls_service_pb2.LayerData(
                        layer_index=layer_idx,
                        parameters=data,
                        shape=shape,
                        dtype=dtype
                    )
                    layer_data_list.append(layer_data)
            
            response = mpls_service_pb2.LayerResponse(
                peer_id=self.worker.config.worker_id,
                layers=layer_data_list,
                iteration=self.worker.current_iteration
            )
            
            logger.debug(f"Worker {self.worker.config.worker_id} sent {len(layer_data_list)} layers")
            return response
            
        except Exception as e:
            logger.error(f"Error in PullLayers: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return mpls_service_pb2.LayerResponse()
    
    def GetMetadata(self, request, context):
        """Handle metadata requests from peers.
        
        Args:
            request: MetadataRequest message
            context: gRPC context
            
        Returns:
            PeerMetadata with worker status and data distribution
        """
        try:
            metadata = mpls_service_pb2.PeerMetadata(
                peer_id=self.worker.config.worker_id,
                data_distribution=self.worker.config.data_distribution.tolist(),
                status=self.worker.status,
                current_iteration=self.worker.current_iteration,
                compute_speed=self.worker.config.compute_speed
            )
            return metadata
            
        except Exception as e:
            logger.error(f"Error in GetMetadata: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return mpls_service_pb2.PeerMetadata()
    
    def Heartbeat(self, request, context):
        """Handle heartbeat requests from peers.
        
        Args:
            request: HeartbeatRequest message
            context: gRPC context
            
        Returns:
            HeartbeatResponse confirming alive status
        """
        response = mpls_service_pb2.HeartbeatResponse(
            peer_id=self.worker.config.worker_id,
            alive=True,
            timestamp=int(time.time() * 1000)
        )
        return response
    
    def MeasureBandwidth(self, request, context):
        """Handle bandwidth measurement requests.
        
        Args:
            request: BandwidthRequest message
            context: gRPC context
            
        Returns:
            BandwidthResponse with test payload
        """
        payload_size = request.payload_size_mb * 1024 * 1024  # Convert to bytes
        payload = b'0' * payload_size
        
        response = mpls_service_pb2.BandwidthResponse(
            peer_id=self.worker.config.worker_id,
            payload=payload,
            timestamp=int(time.time() * 1000)
        )
        return response


class NetworkManager:
    """Manages network communication for a distributed worker."""
    
    def __init__(self, worker, listen_address: str):
        """Initialize network manager.
        
        Args:
            worker: DistributedWorker instance
            listen_address: Address to listen on (e.g., "0.0.0.0:50051")
        """
        self.worker = worker
        self.listen_address = listen_address
        self.serializer = TensorSerializer()
        
        # gRPC server
        self.server = None
        
        # Client stubs (connection pool)
        self.stubs: Dict[int, mpls_service_pb2_grpc.MPLSServiceStub] = {}
        self.channels: Dict[int, grpc.Channel] = {}
        
        # Bandwidth measurements
        self.bandwidth_cache: Dict[int, float] = {}
        
        logger.info(f"NetworkManager initialized for worker {worker.config.worker_id}")
    
    def start_server(self):
        """Start the gRPC server."""
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        mpls_service_pb2_grpc.add_MPLSServiceServicer_to_server(
            MPLSServicer(self.worker), self.server
        )
        self.server.add_insecure_port(self.listen_address)
        self.server.start()
        logger.info(f"Worker {self.worker.config.worker_id} gRPC server started on {self.listen_address}")
    
    def stop_server(self):
        """Stop the gRPC server."""
        if self.server:
            self.server.stop(grace=5)
            logger.info(f"Worker {self.worker.config.worker_id} gRPC server stopped")
    
    def connect_to_peer(self, peer_id: int, peer_address: str):
        """Establish connection to a peer.
        
        Args:
            peer_id: Peer worker ID
            peer_address: Peer address (e.g., "worker1:50051")
        """
        if peer_id in self.channels:
            return  # Already connected
        
        try:
            channel = grpc.insecure_channel(
                peer_address,
                options=[
                    ('grpc.max_send_message_length', 100 * 1024 * 1024),  # 100 MB
                    ('grpc.max_receive_message_length', 100 * 1024 * 1024),
                ]
            )
            stub = mpls_service_pb2_grpc.MPLSServiceStub(channel)
            
            self.channels[peer_id] = channel
            self.stubs[peer_id] = stub
            
            logger.info(f"Worker {self.worker.config.worker_id} connected to peer {peer_id} at {peer_address}")
            
        except Exception as e:
            logger.error(f"Failed to connect to peer {peer_id}: {e}")
    
    def disconnect_from_peer(self, peer_id: int):
        """Disconnect from a peer.
        
        Args:
            peer_id: Peer worker ID
        """
        if peer_id in self.channels:
            self.channels[peer_id].close()
            del self.channels[peer_id]
            del self.stubs[peer_id]
            logger.info(f"Worker {self.worker.config.worker_id} disconnected from peer {peer_id}")
    
    def pull_layers(self, peer_id: int, layer_indices: List[int], timeout: float = 30.0) -> Optional[Dict[int, torch.Tensor]]:
        """Pull specific layers from a peer.
        
        Args:
            peer_id: Peer worker ID
            layer_indices: List of layer indices to pull
            timeout: Request timeout in seconds
            
        Returns:
            Dictionary mapping layer_index -> tensor, or None if failed
        """
        if peer_id not in self.stubs:
            logger.error(f"No connection to peer {peer_id}")
            return None
        
        try:
            start_time = time.time()
            
            request = mpls_service_pb2.LayerRequest(
                requester_id=self.worker.config.worker_id,
                layer_indices=layer_indices,
                iteration=self.worker.current_iteration
            )
            
            response = self.stubs[peer_id].PullLayers(request, timeout=timeout)
            
            # Deserialize layers
            layers = {}
            for layer_data in response.layers:
                tensor = self.serializer.deserialize_tensor(
                    layer_data.parameters,
                    list(layer_data.shape),
                    layer_data.dtype,
                    device=self.worker.device
                )
                layers[layer_data.layer_index] = tensor
            
            # Measure bandwidth
            elapsed = time.time() - start_time
            if elapsed > 0:
                # Estimate data size (rough approximation)
                total_params = sum(t.numel() for t in layers.values())
                size_mb = (total_params * 4) / (1024 * 1024)  # Assume float32
                bandwidth_mbps = size_mb / elapsed
                self.bandwidth_cache[peer_id] = bandwidth_mbps
                
            logger.debug(f"Worker {self.worker.config.worker_id} pulled {len(layers)} layers from peer {peer_id}")
            return layers
            
        except grpc.RpcError as e:
            logger.error(f"gRPC error pulling layers from peer {peer_id}: {e.code()} - {e.details()}")
            return None
        except Exception as e:
            logger.error(f"Error pulling layers from peer {peer_id}: {e}", exc_info=True)
            return None
    
    def get_peer_metadata(self, peer_id: int, timeout: float = 5.0) -> Optional[mpls_service_pb2.PeerMetadata]:
        """Get metadata from a peer.
        
        Args:
            peer_id: Peer worker ID
            timeout: Request timeout in seconds
            
        Returns:
            PeerMetadata message or None if failed
        """
        if peer_id not in self.stubs:
            logger.error(f"No connection to peer {peer_id}")
            return None
        
        try:
            request = mpls_service_pb2.MetadataRequest(
                requester_id=self.worker.config.worker_id
            )
            
            response = self.stubs[peer_id].GetMetadata(request, timeout=timeout)
            return response
            
        except grpc.RpcError as e:
            logger.warning(f"Failed to get metadata from peer {peer_id}: {e.code()}")
            return None
    
    def heartbeat(self, peer_id: int, timeout: float = 2.0) -> bool:
        """Send heartbeat to peer to check if alive.
        
        Args:
            peer_id: Peer worker ID
            timeout: Request timeout in seconds
            
        Returns:
            True if peer is alive, False otherwise
        """
        if peer_id not in self.stubs:
            return False
        
        try:
            request = mpls_service_pb2.HeartbeatRequest(
                sender_id=self.worker.config.worker_id,
                timestamp=int(time.time() * 1000)
            )
            
            response = self.stubs[peer_id].Heartbeat(request, timeout=timeout)
            return response.alive
            
        except grpc.RpcError:
            return False
    
    def measure_bandwidth(self, peer_id: int, payload_size_mb: int = 1) -> Optional[float]:
        """Measure bandwidth to a peer.
        
        Args:
            peer_id: Peer worker ID
            payload_size_mb: Test payload size in MB
            
        Returns:
            Bandwidth in MB/s or None if failed
        """
        if peer_id not in self.stubs:
            return None
        
        try:
            start_time = time.time()
            
            request = mpls_service_pb2.BandwidthRequest(
                requester_id=self.worker.config.worker_id,
                payload_size_mb=payload_size_mb
            )
            
            response = self.stubs[peer_id].MeasureBandwidth(request, timeout=30.0)
            
            elapsed = time.time() - start_time
            if elapsed > 0:
                bandwidth = payload_size_mb / elapsed
                self.bandwidth_cache[peer_id] = bandwidth
                logger.debug(f"Measured bandwidth to peer {peer_id}: {bandwidth:.2f} MB/s")
                return bandwidth
            
        except grpc.RpcError as e:
            logger.warning(f"Failed to measure bandwidth to peer {peer_id}: {e.code()}")
        
        return None
    
    def get_bandwidth(self, peer_id: int) -> float:
        """Get cached or measured bandwidth to a peer.
        
        Args:
            peer_id: Peer worker ID
            
        Returns:
            Bandwidth in MB/s (default 10.0 if unknown)
        """
        if peer_id in self.bandwidth_cache:
            return self.bandwidth_cache[peer_id]
        
        # Try to measure
        bw = self.measure_bandwidth(peer_id)
        return bw if bw is not None else 10.0  # Default bandwidth
    
    def shutdown(self):
        """Shutdown network manager and close all connections."""
        self.stop_server()
        for peer_id in list(self.channels.keys()):
            self.disconnect_from_peer(peer_id)
        logger.info(f"Worker {self.worker.config.worker_id} network manager shutdown complete")
