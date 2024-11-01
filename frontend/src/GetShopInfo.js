import React, { useState } from "react";
import "bootstrap/dist/css/bootstrap.min.css";
import { Modal, Button, Spinner } from "react-bootstrap";
import "./assets/css/shopStyle.css";
import "./assets/css/glitch.css";

const assetUrls = [
  {
    did: "did:op:2a92e2d5ac02dece1946136dfb541807fbf758cd4df9febcf290067a2762b0f2",
    url: "https://market.oceanprotocol.com/asset/did:op:2a92e2d5ac02dece1946136dfb541807fbf758cd4df9febcf290067a2762b0f2",
  },
  {
    did: "did:op:4041a010552158d2343a031579e06074ddd7175a81c32a40e340657d54b0a54d",
    url: "https://market.oceanprotocol.com/asset/did:op:4041a010552158d2343a031579e06074ddd7175a81c32a40e340657d54b0a54d",
  },
];

const AssetCard = ({ asset, url, onPurchaseClick }) => (
  <div className="col-md">
    <div className="card-sl">
      <div className="card-image">
        <img
          src={asset.thumbnail}
          className="card-img-top cardImage"
          alt={asset.name}
          style={{ cursor: "pointer", height: "100px", objectFit: "cover" }}
          onClick={() => window.open(url, "_blank")}
        />
      </div>
      <div className="card-heading">{asset.name}</div>
      <div className="card-text text-justify">{asset.description}</div>
      <div className="card-text">- {asset.author}</div>
      <a className="card-button" onClick={() => onPurchaseClick(asset)}>
        Purchase
      </a>
    </div>
  </div>
);

const GetShopInfo = ({ onPurchaseClick }) => {
  const [assets, setAssets] = useState([]);
  const [modalVisible, setModalVisible] = useState(false);
  const [loading, setLoading] = useState(false);

  const fetchData = async (did) => {
    try {
      const response = await fetch(
        `https://v4.aquarius.oceanprotocol.com/api/aquarius/assets/ddo/${did}`
      );
      const result = await response.json();
      const { metadata, services } = result;
      const thumbnail =
        services[0]?.consumerParameters[0]?.default ||
        "https://via.placeholder.com/150";

      return {
        author: metadata.author || "Unknown",
        created: metadata.created || "Unknown",
        description: metadata.description || "No description available",
        name: metadata.name || "Unnamed Asset",
        thumbnail,
        did,
      };
    } catch (error) {
      console.error("Error fetching data:", error);
    }
  };

  const fetchAllAssets = async () => {
    setLoading(true);
    const fetchedAssets = await Promise.all(
      assetUrls.map(async (asset) => {
        const assetData = await fetchData(asset.did);
        return { ...assetData, url: asset.url };
      })
    );

    setAssets(fetchedAssets.filter((asset) => asset));
    setLoading(false);
  };

  const handleOpenModal = () => {
    setModalVisible(true);
    fetchAllAssets();
  };

  const handlePurchaseClick = (asset) => {
    handleCloseModal();
    onPurchaseClick(asset);
  }

  const handleCloseModal = () => {
    setModalVisible(false);
  };

  return (
    <>
      <button className="ocean-btn" onClick={handleOpenModal}>
        <img
          width="24px"
          className="d-inline"
          src={require("./assets/images/oceanlogo.png")}
          alt="AI Icon"
        />
        <p className="d-inline px-2"> Creativity Market</p>
      </button>

      <Modal show={modalVisible} size="lg" onHide={handleCloseModal}>
        <Modal.Header closeButton>
          <Modal.Title>Creativity Market</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          {loading ? (
            <div className="text-center">
              <Spinner animation="border" role="status" />
              <span className="ms-2">Loading...</span>
            </div>
          ) : (
            <div className="row">
              {assets.map((asset, index) => (
                <AssetCard
                  key={index}
                  asset={asset}
                  url={asset.url}
                  onPurchaseClick={handlePurchaseClick}
                />
              ))}
            </div>
          )}
        </Modal.Body>
        <Modal.Footer>
          <Button variant="secondary" onClick={handleCloseModal}>
            Close
          </Button>
        </Modal.Footer>
      </Modal>
    </>
  );
};

export default GetShopInfo;
