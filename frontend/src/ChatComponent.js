// ChatComponent.js
import React, { Component } from "react";
import "./assets/css/chatstyle.css";

class ChatComponent extends Component {
  constructor(props) {
    super(props);
    this.state = {
      messages: [],
      loading: false,
    };
    this.userMessageRef = React.createRef();
  }

  show_loading_progress = () => {
    this.setState({ loading: true });
  };

  hide_loading_progress = () => {
    this.setState({ loading: false });
  };

  polite_wait = () => new Promise((resolve) => setTimeout(resolve, 300));

  add_user_message = async () => {
    const userMessage = this.userMessageRef.current.value.trim();
    if (!userMessage) return;

    this.setState(
      (prevState) => ({
        messages: [
          ...prevState.messages,
          { text: userMessage, sender: "user" },
        ],
      }),
      this.scrollToBottom
    );

    this.userMessageRef.current.value = "";
    let process_handle = null;
    process_handle = this.props.processMessage(userMessage);

    await this.polite_wait();
    this.show_loading_progress();

    const replay = await process_handle;
    this.hide_loading_progress();

    if (replay !== null && replay.length > 0) this.add_agent_message(replay);
  };

  add_agent_message = (message, imageUrl = null) => {
    this.setState(
      (prevState) => ({
        messages: [
          ...prevState.messages,
          { text: message, sender: "agent", imageUrl },
        ],
      }),
      this.scrollToBottom
    );
  };

  scrollToBottom = () => {
    const chatBox = document.querySelector(".messages-content");
    chatBox.scrollTo({
      top: chatBox.scrollHeight,
      behavior: "smooth",
    });
  };

  render() {
    return (
      <section className="chat">
        <div className="messages">
          <div className="messages-content" style={{ padding: "16px" }}>
            {this.state.messages.map((msg, index) => (
              <div
                key={index}
                className={`message ${
                  msg.sender === "user" ? "message-personal" : "new"
                }`}
              >
                {msg.sender === "agent" && (
                  <figure className="avatar">
                    <img src="assest/images/AIIcon.png" />
                  </figure>
                )}
                {msg.text}
                {msg.imageUrl && (
                  <div className="message-image">
                    <img src={msg.imageUrl} alt="Attached" />
                  </div>
                )}
              </div>
            ))}
            {this.state.loading && (
              <div className="message loading new">
                <figure className="avatar">
                  <img src="assest/images/AIIcon.png" />
                </figure>
                <span></span>
              </div>
            )}
          </div>
        </div>
        <div className="message-box">
          <textarea
            ref={this.userMessageRef}
            className="message-input"
            placeholder="Type message..."
          ></textarea>
          <button onClick={this.add_user_message} className="message-submit">
            Send
          </button>
        </div>
      </section>
    );
  }
}

export default ChatComponent;
