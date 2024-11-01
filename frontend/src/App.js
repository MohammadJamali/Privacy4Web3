import React, { useState, useEffect, useRef } from "react";
import { ethers } from "ethers";
import "bootstrap/dist/css/bootstrap.css";
import "./assets/css/chatstyle.css";
import GetShopInfo from "./GetShopInfo";
import ChatComponent from "./ChatComponent";

import contractABI from "./abis/contractABI.json";

const contractAddress = "0x5FbDB2315678afecb367f032d93F642f64180aa3";
const ProjectName = "Copyright Aware AI";

const App = () => {
  const [web3, setWeb3] = useState(null);
  const [account, setAccount] = useState(null);
  const [plugin, setPlugin] = useState(null);
  const networkName = "OASIS";
  const chatRef = useRef(null);

  const onPurchaseClick = (asset) => {
    chatRef.current.add_agent_message(
      `Nice! From now on, I know that \`${asset.name}\``,
      asset.thumbnail
    );
    setPlugin(asset);
  };

  useEffect(() => {
    if (window.ethereum) {
      setWeb3(new ethers.BrowserProvider(window.ethereum));
    } else {
      console.log("Please install MetaMask!");
    }
  }, []);

  const connectWallet = async () => {
    if (!web3) {
      alert(
        "Web3 provider is not available. Please make sure MetaMask is installed."
      );
      return;
    }

    const provider = new ethers.BrowserProvider(window.ethereum);
    setWeb3(provider);

    try {
      const accounts = await provider.send("eth_requestAccounts", []);
      setAccount(accounts[0]);
    } catch (error) {
      console.error("User denied account access:", error);
    }
  };

  const disconnectWallet = () => {
    setAccount(null);
  };

  const processMessage = async (message) => {
    if (!web3 || !account) {
      alert("Connect your wallet first!");
      return null;
    }

    try {
      const signer = await web3.getSigner();
      const contract = new ethers.Contract(
        contractAddress,
        contractABI,
        signer
      );

      let did = "NULL";
      if (plugin !== null) did = plugin.did;

      // Call the function and await the transaction
      console.log("message:", message, " did:", did);
      const tx = await contract.processPrompt(message, did);
      const receipt = await tx.wait();
      console.log("Transaction successful:", receipt);

      // Step 1: Retrieve message history
      const messageIds = await contract.getUserHistory(account);
      console.log("User's message history IDs:", messageIds);

      if (messageIds.length === 0) {
        console.log("No messages found for this user.");
        return;
      }

      // Step 2: Get the last message ID
      const lastMessageId = messageIds[messageIds.length - 1];
      console.log("Last message ID:", lastMessageId);

      while (true) {
        const [prompt, plugin, reply] = await contract.getMessage(
          lastMessageId
        );
        console.log(
          `Message ID ${lastMessageId} details - Prompt: ${prompt}, Plugin: ${plugin}, Reply: ${reply}`
        );

        if (reply && reply.length > 0) {
          console.log("Reply received:", reply);

          return reply;
        } else {
          console.log("Empty Reply, Waiting ...");
          await new Promise((r) => setTimeout(r, 2000));
        }
      }
    } catch (error) {
      console.error("Error sending message:", error);
      return "null";
    }
  };

  return (
    <div className="fluid">
      <section className="row headnav align-content-center">
        <div
          className="col-sm-6 align-content-start ps-4"
          style={{ justifyItems: "start" }}
        >
          <GetShopInfo onPurchaseClick={onPurchaseClick} />
        </div>

        <div
          className="col-sm-6 pe-4"
          style={{ justifyItems: "end", textAlign: "end" }}
        >
          {account ? (
            <button
              onClick={disconnectWallet}
              className="wallet-btn  disconnect-btn"
            >
              Disconnect
            </button>
          ) : (
            <button onClick={connectWallet} className="wallet-btn connect-btn">
              <img
                width="24px"
                className="d-inline"
                src={require("./assets/images/oasis.png")}
                alt="OASIS Protocol"
              />
              <p className="d-inline px-2">Connect Your Wallet</p>
            </button>
          )}
        </div>
      </section>

      <div className="hero-container">
        <div className="environment"></div>
        <h2 className="hero glitch layers" data-text="{ProjectName}">
          <span>{ProjectName}</span>
        </h2>
      </div>

      <ChatComponent ref={chatRef} processMessage={processMessage} />

      <div className="footer">
        {account ? (
          <span className="useraddress">
            Network: {networkName} - Connected: {account}
          </span>
        ) : (
          <span className="useraddress">
            <a href="https://metamask.io/download/">
              {" "}
              &gt; install metamask ...
            </a>
          </span>
        )}
      </div>
      <div className="bg"></div>
    </div>
  );
};

export default App;
