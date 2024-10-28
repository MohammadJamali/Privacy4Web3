import { ethers } from "hardhat";

describe("Simulate a chat message", () => {
    it("Works", async ()=>{
        const factory = await ethers.getContractFactory('AIChat');
        const contract = await factory.deploy();
        await contract.waitForDeployment();
    });
})