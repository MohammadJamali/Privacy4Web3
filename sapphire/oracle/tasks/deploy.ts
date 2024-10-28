import { task } from "hardhat/config";
import { HardhatRuntimeEnvironment } from "hardhat/types";

task("deploy", "Deploy AiChat Contract").setAction(
  async (args, hre: HardhatRuntimeEnvironment) => {
    const factory = await hre.ethers.getContractFactory("AIChat");
    const contract = await factory.deploy();

    const transaction = await contract.deploymentTransaction();
    const address = await contract.getAddress();

    await contract.waitForDeployment();

    console.log("Contract deployed");
    console.log("Transaction", transaction);
    console.log("Contract Address", address);
  }
);
