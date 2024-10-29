// SPDX-License-Identifier: MIT

pragma solidity ^0.8.0;

import {Subcall} from "@oasisprotocol/sapphire-contracts/contracts/Subcall.sol";

contract AIChat {
    event MessageSent(address indexed user, uint256 messageId);
    event AgentReplied(uint256 indexed messageId, string reply);

    struct Message {
        string prompt;
        string reply;
        string plugin;
        string status;
    }

    bytes21 public roflAppID;
    Message[] private messages; // Stores Message struct by ID
    uint256[] public processingMessages; // Array of message IDs in processing state
    mapping(address => uint256[]) public history; // Holds message IDs for each user

    constructor(bytes21 _roflAppID) {
        roflAppID = _roflAppID;
    }

    function getProcessingMessages() external view returns (uint256[] memory) {
        return processingMessages;
    }

    function processPrompt(
        string memory prompt,
        string memory plugin
    ) external {
        require(
            bytes(prompt).length > 0 && bytes(prompt).length <= 280,
            "Invalid prompt"
        );

        uint256 messageId = messages.length;
        messages.push(
            Message({
                prompt: prompt,
                reply: "",
                plugin: plugin,
                status: "Processing"
            })
        );

        history[msg.sender].push(messageId);
        processingMessages.push(messageId);

        emit MessageSent(msg.sender, messageId);
    }

    function submitAgentReply(uint256 messageId, string memory reply) external {
        // Subcall.roflEnsureAuthorizedOrigin(roflAppID);

        require(messageId < messages.length, "Invalid message ID");

        messages[messageId].status = "Done";
        messages[messageId].reply = reply;

        for (uint256 i = 0; i < processingMessages.length; i++) {
            if (processingMessages[i] == messageId) {
                processingMessages[i] = processingMessages[
                    processingMessages.length - 1
                ];
                processingMessages.pop();
                break;
            }
        }

        emit AgentReplied(messageId, reply);
    }

    function getMessage(
        uint256 messageId
    ) external view returns (string memory, string memory, string memory) {
        require(messageId < messages.length, "Invalid message ID");

        return (
            messages[messageId].prompt,
            messages[messageId].plugin,
            messages[messageId].reply
        );
    }

    function getUserHistory(
        address user
    ) external view returns (uint256[] memory) {
        return history[user];
    }
}
