import { Immutable, MessageEvent, PanelExtensionContext, Topic } from "@foxglove/extension";
import React, { FormEvent, ReactElement, useEffect, useLayoutEffect, useState } from "react";
import { createRoot } from "react-dom/client";

type PressedKeys = {
  a: boolean;
  w: boolean;
  s: boolean;
  d: boolean;
  " ": boolean;
};

function toBitmap(pressedKeys: PressedKeys): number {
  return (
    (pressedKeys.w ? 1 : 0) +
    (pressedKeys.a ? 2 : 0) +
    (pressedKeys.s ? 4 : 0) +
    (pressedKeys.d ? 8 : 0) +
    (pressedKeys[" "] ? 16 : 0)
  );
}

function ParentPanel({ context }: { context: PanelExtensionContext }): ReactElement {
  const [inputValue, setInputValue] = useState("");
  const [nickname, setNickname] = useState<undefined | string>(undefined);

  const handleSubmit = (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault(); // prevent page reload
    setNickname(inputValue);
    setInputValue(""); // clear input if desired
  };

  return nickname == undefined ? (
    <>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          placeholder="Enter your name"
          value={inputValue}
          onChange={(e) => {
            setInputValue(e.target.value);
          }}
        />
        <button type="submit">Submit</button>
      </form>
    </>
  ) : (
    <KeyListenPanel context={context} nickname={nickname} />
  );
}

function KeyListenPanel({
  context,
  nickname,
}: {
  context: PanelExtensionContext;
  nickname: string;
}): ReactElement {
  const [topics, setTopics] = useState<undefined | Immutable<Topic[]>>();
  const [messages, setMessages] = useState<undefined | Immutable<MessageEvent[]>>();
  const [pressedKeys, setPressedKeys] = useState<PressedKeys>({
    a: false,
    w: false,
    s: false,
    d: false,
    " ": false,
  });
  const [canPublish, setCanPublish] = useState<boolean | undefined>(undefined);

  const [renderDone, setRenderDone] = useState<(() => void) | undefined>();

  // We use a layout effect to setup render handling for our panel. We also setup some topic subscriptions.
  useLayoutEffect(() => {
    // The render handler is run by the broader studio system during playback when your panel
    // needs to render because the fields it is watching have changed. How you handle rendering depends on your framework.
    // You can only setup one render handler - usually early on in setting up your panel.
    //
    // Without a render handler your panel will never receive updates.
    //
    // The render handler could be invoked as often as 60hz during playback if fields are changing often.
    context.onRender = (renderState, done) => {
      // render functions receive a _done_ callback. You MUST call this callback to indicate your panel has finished rendering.
      // Your panel will not receive another render callback until _done_ is called from a prior render. If your panel is not done
      // rendering before the next render call, studio shows a notification to the user that your panel is delayed.
      //
      // Set the done callback into a state variable to trigger a re-render.
      setRenderDone(() => done);

      // We may have new topics - since we are also watching for messages in the current frame, topics may not have changed
      // It is up to you to determine the correct action when state has not changed.
      setTopics(renderState.topics);

      // currentFrame has messages on subscribed topics since the last render call
      setMessages(renderState.currentFrame);
    };

    // After adding a render handler, you must indicate which fields from RenderState will trigger updates.
    // If you do not watch any fields then your panel will never render since the panel context will assume you do not want any updates.

    // tell the panel context that we care about any update to the _topic_ field of RenderState
    context.watch("topics");

    // tell the panel context we want messages for the current frame for topics we've subscribed to
    // This corresponds to the _currentFrame_ field of render state.
    context.watch("currentFrame");

    // subscribe to some topics, you could do this within other effects, based on input fields, etc
    // Once you subscribe to topics, currentFrame will contain message events from those topics (assuming there are messages).
    context.subscribe([{ topic: "/server-pulse" }]);
    if (context.advertise) {
      context.advertise("/keys", "number");
      context.advertise("/my-name-is", "string");
      setCanPublish(true);
    } else {
      setCanPublish(false);
    }
  }, [context]);

  // Set up keyboard event listeners
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      const key = event.key.toLowerCase();
      if (key in pressedKeys) {
        setPressedKeys((pk) => ({ ...pk, [key]: true }));
      }
    };

    const handleKeyUp = (event: KeyboardEvent) => {
      const key = event.key.toLowerCase();
      if (key in pressedKeys) {
        setPressedKeys((pk) => ({ ...pk, [key]: false }));
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    // Clean up the event listener on unmount
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
    };
  });

  useEffect(() => {
    if (canPublish != undefined && canPublish) {
      const bitmap = toBitmap(pressedKeys);
      context.publish!("/keys", bitmap);
      context.publish!("/my-name-is", nickname);
    }
  }, [canPublish, pressedKeys, context, nickname]);

  // invoke the done callback once the render is complete
  useEffect(() => {
    renderDone?.();
  }, [renderDone]);

  return (
    <div style={{ padding: "1rem" }}>
      <div
        style={{
          marginTop: "1rem",
          padding: "0.5rem",
          border: "1px solid #ccc",
          borderRadius: "4px",
        }}
      >
        {canPublish != undefined && canPublish ? "can publish" : "cannot publish"}
        <h3>Keyboard Input</h3>
        <p>
          pressed keys:{" "}
          <strong>
            [
            {Object.entries(pressedKeys)
              .filter(([_, value]) => value)
              .map(([key]) => key)
              .join(", ")}
            ]
          </strong>
        </p>
        <p>
          <em>Press W, A, S, D, or space to see it displayed here</em>
        </p>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          rowGap: "0.2rem",
          marginTop: "1rem",
        }}
      >
        <b style={{ borderBottom: "1px solid" }}>Topic</b>
        <b style={{ borderBottom: "1px solid" }}>Schema name</b>
        {(topics ?? []).map((topic) => (
          <>
            <div key={topic.name}>{topic.name}</div>
            <div key={topic.schemaName}>{topic.schemaName}</div>
          </>
        ))}
      </div>
      <div>{messages?.length}</div>
    </div>
  );
}

export function initExamplePanel(context: PanelExtensionContext): () => void {
  const root = createRoot(context.panelElement);
  root.render(<ParentPanel context={context} />);

  // Return a function to run when the panel is removed
  return () => {
    root.unmount();
  };
}
