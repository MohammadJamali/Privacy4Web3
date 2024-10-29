import { bech32 } from "bech32";
import { task } from "hardhat/config";
import { HardhatRuntimeEnvironment } from "hardhat/types";

task("deploy", "Deploy AiChat Contract")
  .addPositionalParam("roflAppID", "ROFL App ID in Bech32 format")
  .setAction(async ({ roflAppID }, hre: HardhatRuntimeEnvironment) => {
    const { prefix, words } = bech32.decode(roflAppID);

    const factory = await hre.ethers.getContractFactory("AIChat");
    const contract = await factory.deploy(
      new Uint8Array(bech32.fromWords(words))
    );

    const transaction = await contract.deploymentTransaction();
    const contractAddress = await contract.getAddress();

    await contract.waitForDeployment();

    console.log("Contract deployed");
    console.log("Transaction", transaction);
    console.log(
      `Oracle for ROFL app ${roflAppID} deployed to ${contractAddress}`
    );
  });