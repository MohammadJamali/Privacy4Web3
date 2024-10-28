import { hexlify } from "ethers";
import { randomBytes } from "ethers/crypto";
import { artifacts, ethers } from "hardhat";
import { expect } from "chai";
import { AIChat } from "../typechain-types";

describe("AIChat Contract", () => {
  const prompt = "Hello, AI!";
  const plugin = "examplePlugin";

  let contract: AIChat, owner, user1, user2;
  const fakeRoflAppID = hexlify(randomBytes(21));
  console.log("Generated Fake RoFL Id", fakeRoflAppID);

  beforeEach(async () => {
    const factory = await ethers.getContractFactory("AIChat");
    contract = await factory.deploy(fakeRoflAppID);
    await contract.waitForDeployment();

    [owner, user1, user2] = await ethers.getSigners();
  });

  describe("Deployment", () => {
    it("Should set the correct RoflAppID", async () => {
      expect(await contract.roflAppID()).to.equal(fakeRoflAppID);
    });
  });

  describe("processPrompt", () => {
    it("Should create a new message and emit MessageSent event", async () => {
      await expect(contract.connect(user1).processPrompt(prompt, plugin))
        .to.emit(contract, "MessageSent")
        .withArgs(user1.address, 0);

      const messageCount = await contract.messageCount(user1.address);
      expect(messageCount).to.equal(1);

      const message = await contract.history(user1.address, 0);
      expect(message.prompt).to.equal(prompt);
      expect(message.plugin).to.equal(plugin);
      expect(message.status).to.equal("Processing");
    });

    it("Should reject prompts longer than 280 characters", async () => {
      const longPrompt = "A".repeat(281);
      const plugin = "examplePlugin";

      await expect(
        contract.connect(user1).processPrompt(longPrompt, plugin)
      ).to.be.revertedWith("Invalid prompt");
    });

    it("Should allow an agent to reply and emit AgentReplied event", async () => {
      const reply = "Agent's response";

      await contract
        .connect(owner)
        .submitAgentReply(user1.address, 0, reply);

      const message = await contract.history(user1.address, 0);
      expect(message.reply).to.equal(reply);

      await expect(
        contract.connect(owner).submitAgentReply(user1.address, 0, reply)
      )
        .to.emit(contract, "AgentReplied")
        .withArgs(user1.address, 0, reply);
    });

    it("Should reject invalid message IDs", async () => {
      await expect(
        contract
          .connect(owner)
          .submitAgentReply(user1.address, 1, "Another reply")
      ).to.be.revertedWith("Invalid message ID");
    });

    it("Should return the correct message count for a user", async () => {
      await contract.connect(user1).processPrompt(prompt, plugin);
      await contract.connect(user1).processPrompt("Another message", "pluginB");

      const messageCount = await contract.connect(user1).messages();
      expect(messageCount).to.equal(2);
    });

    it("Should return the correct message details for valid message ID", async () => {
      const anotherPrompt = "Another message";
      const anothereply = "Another message";
      const anotherPlugin = "pluginB";

      await contract.connect(user1).processPrompt(prompt, plugin);
      await contract
        .connect(owner)
        .submitAgentReply(user1.address, 0, anothereply);
      await contract.connect(user1).processPrompt(anotherPrompt, anotherPlugin);

      const message = await contract.connect(user1).message(user1.address, 0);

      expect(message.prompt).to.equal(prompt);
      expect(message.reply).to.equal(anothereply);
      expect(message.plugin).to.equal(plugin);
      expect(message.status).to.equal("Done");

      const anotherMessage = await contract
        .connect(user1)
        .message(user1.address, 1);

      expect(anotherMessage.prompt).to.equal(anotherPrompt);
      expect(anotherMessage.plugin).to.equal(anotherPlugin);
      expect(anotherMessage.status).to.equal("Processing");
    });

    it("Should reject access to invalid message IDs", async () => {
      await expect(
        contract.connect(user1).message(user1.address, 1)
      ).to.be.revertedWith("Invalid message ID");
    });
  });
});
