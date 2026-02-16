"""
Module for interacting with VTube Studio API via WebSocket.
Handles authentication and parameter injection.
"""

import asyncio
import websockets
import json
import logging
import os
from typing import Dict, Optional
import uuid

from src.optimized_parameter_mapper import optimized_mapper

logger = logging.getLogger(__name__)


class VTubeStudioClient:
    """
    Client for VTube Studio WebSocket API.
    """

    def __init__(self, host: str = "localhost", port: int = 8001):
        """
        Initialize VTube Studio client.

        Args:
            host: VTube Studio host.
            port: VTube Studio WebSocket port.
        """
        self.uri = f"ws://{host}:{port}"
        self.websocket = None
        self.auth_token: Optional[str] = None
        self.plugin_name = "MediaPipeLive2D"
        self.plugin_developer = "User"
        self.host = host
        self.port = port
        self.is_connected = False
        self.is_authenticated = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.base_reconnect_delay = 1.0  # seconds
        self.max_reconnect_delay = 60.0  # seconds
        self._load_auth_token()

    def _save_auth_token(self):
        """Save authentication token to file for persistent storage."""
        if self.auth_token:
            try:
                with open("vtube_auth_token.json", "w") as f:
                    json.dump({"auth_token": self.auth_token}, f)
                logger.info("Authentication token saved to file")
            except Exception as e:
                logger.warning(f"Could not save authentication token: {e}")

    def _load_auth_token(self):
        """Load authentication token from file if it exists."""
        try:
            if os.path.exists("vtube_auth_token.json"):
                with open("vtube_auth_token.json", "r") as f:
                    data = json.load(f)
                    self.auth_token = data.get("auth_token")
                    if self.auth_token:
                        logger.info("Loaded previous authentication token from file")
        except Exception as e:
            logger.warning(f"Could not load authentication token: {e}")

    async def connect(self) -> bool:
        """
        Connect to VTube Studio WebSocket server.

        Returns:
            True if connection established, False otherwise.
        """
        try:
            self.websocket = await websockets.connect(self.uri)
            self.is_connected = True
            self.reconnect_attempts = 0  # Reset reconnect attempts on successful connection
            logger.info("Connected to VTube Studio at %s", self.uri)
            return True
        except Exception as e:
            logger.error("Failed to connect to VTube Studio: %s", e)
            self.is_connected = False
            self.is_authenticated = False
            return False

    async def authenticate(self) -> bool:
        """
        Authenticate with VTube Studio API.

        Returns:
            True if authentication successful, False otherwise.
        """
        if not self.websocket:
            logger.error("Not connected to VTube Studio")
            return False

        # Try to use existing token first
        if self.auth_token:
            result = await self._send_authentication_request(self.auth_token)
            self.is_authenticated = result
            if result:
                self._save_auth_token()  # Save token if authentication successful
            return result

        # Request new token
        request = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": str(uuid.uuid4())[:10],
            "messageType": "AuthenticationTokenRequest",
            "data": {
                "pluginName": self.plugin_name,
                "pluginDeveloper": self.plugin_developer
            }
        }

        try:
            await self.websocket.send(json.dumps(request))
            response = await self.websocket.recv()
            response_data = json.loads(response)

            if response_data.get("messageType") == "AuthenticationTokenResponse":
                self.auth_token = response_data["data"]["authenticationToken"]
                logger.info("Received authentication token")
                self._save_auth_token()  # Save the new token
                # Now authenticate with the token
                result = await self._send_authentication_request(self.auth_token)
                self.is_authenticated = result
                return result
            else:
                error = response_data.get("data", {}).get("message", "Unknown error")
                logger.error("Failed to get authentication token: %s", error)
                self.is_authenticated = False
                return False

        except Exception as e:
            logger.error("Authentication error: %s", e)
            self.is_authenticated = False
            return False

    async def _send_authentication_request(self, token: str) -> bool:
        """
        Send authentication request with token.

        Args:
            token: Authentication token.

        Returns:
            True if authentication successful.
        """
        request = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": str(uuid.uuid4())[:10],
            "messageType": "AuthenticationRequest",
            "data": {
                "pluginName": self.plugin_name,
                "pluginDeveloper": self.plugin_developer,
                "authenticationToken": token
            }
        }

        try:
            await self.websocket.send(json.dumps(request))
            response = await self.websocket.recv()
            response_data = json.loads(response)

            if response_data.get("messageType") == "AuthenticationResponse":
                authenticated = response_data["data"].get("authenticated", False)
                if authenticated:
                    logger.info("Successfully authenticated with VTube Studio")
                    return True
                else:
                    reason = response_data["data"].get("reason", "Unknown reason")
                    logger.error("Authentication failed: %s", reason)
                    return False
            else:
                error = response_data.get("data", {}).get("message", "Unknown error")
                logger.error("Authentication request failed: %s", error)
                return False

        except Exception as e:
            logger.error("Authentication request error: %s", e)
            return False

    async def create_parameter(self, param_name: str, min_value: float = -10, max_value: float = 10, default_value: float = 0) -> bool:
        """
        Create a new custom tracking parameter in VTube Studio.

        Args:
            param_name: Name of the parameter to create.
            min_value: Minimum value for the parameter.
            max_value: Maximum value for the parameter.
            default_value: Default value for the parameter.

        Returns:
            True if parameter creation was successful.
        """
        if not self.websocket:
            logger.error("Not connected to VTube Studio")
            return False
            
        request = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": str(uuid.uuid4())[:10],
            "messageType": "ParameterCreationRequest",
            "data": {
                "parameterName": param_name,
                "min": min_value,
                "max": max_value,
                "defaultValue": default_value
            }
        }

        try:
            await self.websocket.send(json.dumps(request))
            response = await self.websocket.recv()
            response_data = json.loads(response)
            if response_data.get("messageType") == "APIError":
                error_message = response_data.get("data", {}).get("message", "Unknown error")
                # Ignore error if parameter already exists
                if "already exists" in error_message:
                    logger.info(f"Parameter {param_name} already exists, which is fine.")
                    return True
                logger.error(f"Failed to create parameter {param_name}: {error_message}")
                return False
            logger.info(f"Successfully created parameter {param_name}")
            return True
        except Exception as e:
            logger.error(f"Exception while creating parameter {param_name}: {e}")
            return False

    async def inject_parameters(self, parameters: Dict[str, float], face_found: bool = True) -> bool:
        """
        Inject parameter data to VTube Studio with retry mechanism.

        Args:
            parameters: Dictionary of parameter names to their values.
            face_found: Whether face is currently detected.

        Returns:
            True if injection successful.
        """
        if not self.websocket or not self.is_connected:
            logger.error("Not connected to VTube Studio")
            return False
            
        # Handle empty parameters case
        if not parameters:
            logger.warning("No parameters provided for injection")
            return False

        # Prepare parameter values for injection
        parameter_values = [
            {"id": name, "value": float(value)}
            for name, value in parameters.items()
        ]

        request = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": str(uuid.uuid4())[:10],
            "messageType": "InjectParameterDataRequest",
            "data": {
                "faceFound": face_found,
                "mode": "set",
                "parameterValues": parameter_values
            }
        }

        # Try to inject parameters with retry mechanism
        max_retries = 3
        for attempt in range(max_retries):
            try:
                await self.websocket.send(json.dumps(request))
                # Wait for response to catch immediate errors
                response = await self.websocket.recv()
                response_data = json.loads(response)
                if response_data.get("messageType") == "APIError":
                    error_message = response_data.get("data", {}).get("message", "Unknown error")
                    logger.error("API Error when injecting parameters: %s", error_message)
                    # Check if it's a connection error that requires reconnection
                    if "connection" in error_message.lower() or "disconnected" in error_message.lower():
                        self.is_connected = False
                        self.is_authenticated = False
                        # Try to reconnect if this is not the last attempt
                        if attempt < max_retries - 1:
                            logger.info(f"Attempting to reconnect (attempt {attempt + 1}/{max_retries - 1})")
                            if await self.reconnect():
                                continue  # Retry after reconnection
                    return False
                return True
            except websockets.exceptions.ConnectionClosed:
                logger.error("Connection to VTube Studio was closed")
                self.is_connected = False
                self.is_authenticated = False
                # Try to reconnect if this is not the last attempt
                if attempt < max_retries - 1:
                    logger.info(f"Attempting to reconnect (attempt {attempt + 1}/{max_retries - 1})")
                    if await self.reconnect():
                        continue  # Retry after reconnection
                return False
            except Exception as e:
                logger.error("Failed to inject parameters: %s", e)
                # Check if it's a connection error
                if "connection" in str(e).lower() or "disconnected" in str(e).lower():
                    self.is_connected = False
                    self.is_authenticated = False
                    # Try to reconnect if this is not the last attempt
                    if attempt < max_retries - 1:
                        logger.info(f"Attempting to reconnect (attempt {attempt + 1}/{max_retries - 1})")
                        if await self.reconnect():
                            continue  # Retry after reconnection
                return False
        
        
        return False  # All retry attempts failed

    async def flush_remaining_parameters(self) -> bool:
        """
        Flush any remaining parameters in the optimized mapper batch queue.
        
        Returns:
            True if flush successful or no parameters to flush, False on error.
        """
        # Get any remaining parameters from the optimized mapper
        remaining_params = optimized_mapper.flush_batch()
        
        # If there are remaining parameters, inject them
        if remaining_params:
            logger.info(f"Flushing {len(remaining_params)} remaining parameters")
            return await self.inject_parameters(remaining_params, face_found=True)
        
        return True

    async def close(self):
        """
        Close the WebSocket connection.
        """
        # Flush any remaining parameters before closing
        try:
            await self.flush_remaining_parameters()
        except Exception as e:
            logger.warning(f"Error flushing remaining parameters: {e}")
        
        if self.websocket:
            try:
                await self.websocket.close()
                logger.info("VTube Studio connection closed")
            except Exception as e:
                logger.error("Error while closing VTube Studio connection: %s", e)
        self.websocket = None
        self.is_connected = False
        self.is_authenticated = False
    async def reconnect(self) -> bool:
        """
        Attempt to reconnect to VTube Studio with exponential backoff.
        
        Returns:
            True if reconnection and re-authentication successful, False otherwise.
        """
        logger.info("Attempting to reconnect to VTube Studio...")
        
        # Implement exponential backoff
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error("Maximum reconnection attempts reached (%d)", self.max_reconnect_attempts)
            return False
            
        # Calculate delay with exponential backoff
        delay = min(
            self.base_reconnect_delay * (2 ** self.reconnect_attempts),
            self.max_reconnect_delay
        )
        
        logger.info("Waiting %.1f seconds before reconnection attempt %d/%d", 
                   delay, self.reconnect_attempts + 1, self.max_reconnect_attempts)
        
        # Wait for the calculated delay
        await asyncio.sleep(delay)
        
        # Close existing connection if any
        if self.websocket:
            try:
                await self.websocket.close()
            except:
                pass
            self.websocket = None
            
        # Increment reconnect attempts
        self.reconnect_attempts += 1
            
        # Reconnect
        if await self.connect():
            # Re-authenticate
            if await self.authenticate():
                logger.info("Successfully reconnected and re-authenticated with VTube Studio")
                self.reconnect_attempts = 0  # Reset on successful reconnection
                return True
            else:
                logger.error("Reconnection successful but re-authentication failed")
                return False
        else:
            logger.error("Failed to reconnect to VTube Studio")
            return False
