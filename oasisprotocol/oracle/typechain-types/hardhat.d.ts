/* Autogenerated file. Do not edit manually. */
/* tslint:disable */
/* eslint-disable */

import { ethers } from "ethers";
import {
  DeployContractOptions,
  FactoryOptions,
  HardhatEthersHelpers as HardhatEthersHelpersBase,
} from "@nomicfoundation/hardhat-ethers/types";

import * as Contracts from ".";

declare module "hardhat/types/runtime" {
  interface HardhatEthersHelpers extends HardhatEthersHelpersBase {
    getContractFactory(
      name: "Subcall",
      signerOrOptions?: ethers.Signer | FactoryOptions
    ): Promise<Contracts.Subcall__factory>;
    getContractFactory(
      name: "AIChat",
      signerOrOptions?: ethers.Signer | FactoryOptions
    ): Promise<Contracts.AIChat__factory>;

    getContractAt(
      name: "Subcall",
      address: string | ethers.Addressable,
      signer?: ethers.Signer
    ): Promise<Contracts.Subcall>;
    getContractAt(
      name: "AIChat",
      address: string | ethers.Addressable,
      signer?: ethers.Signer
    ): Promise<Contracts.AIChat>;

    deployContract(
      name: "Subcall",
      signerOrOptions?: ethers.Signer | DeployContractOptions
    ): Promise<Contracts.Subcall>;
    deployContract(
      name: "AIChat",
      signerOrOptions?: ethers.Signer | DeployContractOptions
    ): Promise<Contracts.AIChat>;

    deployContract(
      name: "Subcall",
      args: any[],
      signerOrOptions?: ethers.Signer | DeployContractOptions
    ): Promise<Contracts.Subcall>;
    deployContract(
      name: "AIChat",
      args: any[],
      signerOrOptions?: ethers.Signer | DeployContractOptions
    ): Promise<Contracts.AIChat>;

    // default types
    getContractFactory(
      name: string,
      signerOrOptions?: ethers.Signer | FactoryOptions
    ): Promise<ethers.ContractFactory>;
    getContractFactory(
      abi: any[],
      bytecode: ethers.BytesLike,
      signer?: ethers.Signer
    ): Promise<ethers.ContractFactory>;
    getContractAt(
      nameOrAbi: string | any[],
      address: string | ethers.Addressable,
      signer?: ethers.Signer
    ): Promise<ethers.Contract>;
    deployContract(
      name: string,
      signerOrOptions?: ethers.Signer | DeployContractOptions
    ): Promise<ethers.Contract>;
    deployContract(
      name: string,
      args: any[],
      signerOrOptions?: ethers.Signer | DeployContractOptions
    ): Promise<ethers.Contract>;
  }
}
