"""
Module for interacting with VTube Studio API via WebSocket.
Handles authentication and parameter injection.
"""

import asyncio
import websockets
import json
import logging
from typing import Dict, Any, Optional
import uuid

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
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.auth_token: Optional[str] = None
        self.plugin_name = "MediaPipeLive2D"
        self.plugin_developer = "User"

    async def connect(self) -> bool:
        """
        Connect to VTube Studio WebSocket server.

        Returns:
            True if connection established, False otherwise.
        """
        try:
            self.websocket = await websockets.connect(self.uri)
            logger.info("Connected to VTube Studio at %s", self.uri)
            return True
        except Exception as e:
            logger.error("Failed to connect to VTube Studio: %s", e)
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
            return await self._send_authentication_request(self.auth_token)

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
                # Now authenticate with the token
                return await self._send_authentication_request(self.auth_token)
            else:
                error = response_data.get("data", {}).get("message", "Unknown error")
                logger.error("Failed to get authentication token: %s", error)
                return False

        except Exception as e:
            logger.error("Authentication error: %s", e)
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
        Inject parameter data to VTube Studio.

        Args:
            parameters: Dictionary of parameter names to their values.
            face_found: Whether face is currently detected.

        Returns:
            True if injection successful.
        """
        if not self.websocket:
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

        try:
            await self.websocket.send(json.dumps(request))
            # Wait for response to catch immediate errors
            response = await self.websocket.recv()
            response_data = json.loads(response)
            if response_data.get("messageType") == "APIError":
                error_message = response_data.get("data", {}).get("message", "Unknown error")
                logger.error("API Error when injecting parameters: %s", error_message)
                return False
            return True
        except Exception as e:
            logger.error("Failed to inject parameters: %s", e)
            return False

    async def close(self):
        """
        Close the WebSocket connection.
        """
        if self.websocket:
            try:
                await self.websocket.close()
                logger.info("VTube Studio connection closed")
            except Exception as e:
                logger.error("Error while closing VTube Studio connection: %s", e)
