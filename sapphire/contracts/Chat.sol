// SPDX-License-Identifier: MIT

pragma solidity ^0.8.0;

import {Subcall} from "@oasisprotocol/sapphire-contracts/contracts/Subcall.sol";

contract AIChat {
    event MessageSent(address indexed user, uint256 messageId);
    event AgentReplied(
        address indexed user,
        uint256 indexed messageId,
        string reply
    );

    struct Message {
        string prompt;
        string reply;
        string plugin;
        string status;
    }

    bytes21 public roflAppID;
    mapping(address => mapping(uint256 => Message)) public history;
    mapping(address => uint256) public messageCount;

    constructor(bytes21 _roflAppID) {
        roflAppID = _roflAppID;
    }

    function processPrompt(
        string memory prompt,
        string memory plugin
    ) external {
        require(
            bytes(prompt).length > 0 && bytes(prompt).length <= 280,
            "Invalid prompt"
        );

        uint256 messageId = messageCount[msg.sender]++;
        history[msg.sender][messageId] = Message({
            prompt: prompt,
            reply: "",
            plugin: plugin,
            status: "Processing"
        });

        emit MessageSent(msg.sender, messageId);
    }

    function submitAgentReply(
        address userId,
        uint256 messageId,
        string memory reply
    ) external {
        // Subcall.roflEnsureAuthorizedOrigin(roflAppID);

        require(messageId <= messageCount[userId], "Invalid message ID");

        history[userId][messageId].status = "Done";
        history[userId][messageId].reply = reply;

        emit AgentReplied(userId, messageId, reply);
    }

    function messages() external view returns (uint256) {
        return messageCount[msg.sender];
    }

    function message(
        address userId,
        uint256 messageId
    ) external view returns (Message memory) {
        // if (userId != msg.sender) {
        //     Subcall.roflEnsureAuthorizedOrigin(roflAppID);
        // }

        require(messageId < messageCount[userId], "Invalid message ID");

        return history[userId][messageId];
    }
}
