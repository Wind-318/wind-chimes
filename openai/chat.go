// @file chat.go
// @brief Chat API implementation for OpenAI GPT-3.5 API. (https://beta.openai.com/docs/api-reference/chat)

// Copyright (c) 2023 Wind. All rights reserved.
// Use of this source code is governed by a MIT license
// that can be found in the LICENSE file.

// Package openai is used to call the api of chatgpt-3.5
package openai

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
)

// Message is the message struct.
type Message struct {
	// Role is the role of the message. Can be "user", "system" or "assistant".
	Role string `json:"role"`
	// Content is the content of the message.
	Content string `json:"content"`
}

// Usage is the usage object is used to represent the usage of the API.
type Usage struct {
	// PromptTokens is the number of tokens used for the prompt.
	PromptTokens int `json:"prompt_tokens"`
	// CompletionTokens is the number of tokens used for the completion.
	CompletionTokens int `json:"completion_tokens"`
	// TotalTokens is the total number of tokens used.
	TotalTokens int `json:"total_tokens"`
}

// Choice is the choice object is used to represent a choice in a chat completion.
type Choice struct {
	// The index of the choice.
	Index int `json:"index"`
	// Msg is the message object is used to represent a message in a conversation.
	Msg Message `json:"message"`
	// FinishReason is the reason the chat completion stopped.
	FinishReason string `json:"finish_reason"`
}

// ChatResponse is the chat completion object is used to represent a chat completion.
type ChatResponse struct {
	// ID is the ID of the chat completion.
	ID string `json:"id"`
	// Object is the object type of the chat completion.
	Object string `json:"object"`
	// Created is the timestamp of when the chat completion was created.
	Created int `json:"created"`
	// Model is the ID of the model used to generate the chat completion.
	Choices []Choice `json:"choices"`
	// Usage is the usage object is used to represent the usage of the API.
	Usages Usage `json:"usage"`
}

// Chat is the chat data
type Chat struct {
	// Request data
	data sync.Map
	// Secret key
	key atomic.Value
	// Mutex
	mutex sync.RWMutex
}

// SetAuthorizationKey is used to set authorization key
func (c *Chat) SetAuthorizationKey(key string) {
	c.key.Store(key)
	c.data.Store("model", "gpt-3.5-turbo")
}

func (c *Chat) addMessage(role, content string) {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	if _, ok := c.data.Load("messages"); !ok {
		c.data.Store("messages", []map[string]string{})
	}
	val, _ := c.data.Load("messages")
	var messages []map[string]string = val.([]map[string]string)
	messages = append(messages, map[string]string{
		"role":    role,
		"content": content,
	})
	c.data.Store("messages", messages)
}

// AddMessage is used to add message to the chat.
// The role can be "user", "system" or "assistant".
func (c *Chat) AddMessageAsUser(content string) {
	c.addMessage("user", content)
}

func (c *Chat) AddMessageAsSystem(content string) {
	c.addMessage("system", content)
}

func (c *Chat) AddMessageAsAssistant(content string) {
	c.addMessage("assistant", content)
}

// SetTemperature temperature number Optional Defaults to 1;
// What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random
// while lower values like 0.2 will make it more focused and deterministic.
// We generally recommend altering this or top_p but not both.
func (c *Chat) SetTemperature(temperature float64) {
	c.data.Store("temperature", temperature)
}

// SetTopP top.p number Optional Defaults to 1;
// An alternative to sampling with temperature, called nucleus sampling,
// where the model considers the results of the tokens with top_p probability mass.
// So 0.1 means only the tokens comprising the top 10% probability mass are considered.
// We generally recommend altering this or temperature but not both.
func (c *Chat) SetTopP(topP float64) {
	c.data.Store("top_p", topP)
}

// SetN How many chat completion choices to generate for each input message.
func (c *Chat) SetN(n int) {
	c.data.Store("n", n)
}

// SetStream stream boolean Optional Defaults to false.
// If set, partial message deltas will be sent, like in ChatGPT.
// Tokens will be sent as data-only server-sent events as they become available,
// with the stream terminated by a data: [DONE] message.
func (c *Chat) SetStream(stream bool) {
	c.data.Store("stream", stream)
}

// SetStopStr stop string or array Optional Defaults to null;
// Up to 4 sequences where the API will stop generating further tokens.
func (c *Chat) SetStopStr(stop string) {
	c.data.Store("stop", stop)
}

// SetStopArr stop string or array Optional Defaults to null;
// Up to 4 sequences where the API will stop generating further tokens.
func (c *Chat) SetStopArr(stop []string) {
	c.data.Store("stop", stop)
}

// SetMaxTokens max_tokens integer Optional Defaults to inf;
// The maximum number of tokens allowed for the generated answer.
// By default, the number of tokens the model can return will be (4096 - prompt tokens).
func (c *Chat) SetMaxTokens(maxTokens int) {
	c.data.Store("max_tokens", maxTokens)
}

// SetPresencePenalty presence_penalty number Optional Defaults to 0;
// Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear
// in the text so far, increasing the model's likelihood to talk about new topics.
func (c *Chat) SetPresencePenalty(presencePenalty float64) {
	c.data.Store("presence_penalty", presencePenalty)
}

// SetFrequencyPenalty frequency_penalty number Optional Defaults to 0;
// Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing
// frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
func (c *Chat) SetFrequencyPenalty(frequencyPenalty float64) {
	c.data.Store("frequency_penalty", frequencyPenalty)
}

// SetLogitBias logit_bias map Optional Defaults to null;
// Modify the likelihood of specified tokens appearing in the completion.
// Accepts a json object that maps tokens (specified by their token ID in the tokenizer)
// to an associated bias value from -100 to 100. Mathematically, the bias is added to the logits
// generated by the model prior to sampling. The exact effect will vary per model, but values
// between -1 and 1 should decrease or increase likelihood of selection; values like -100 or 100
// should result in a ban or exclusive selection of the relevant token.
func (c *Chat) SetLogitBias(logitBias map[string]int) {
	c.data.Store("logit_bias", logitBias)
}

// SetUser user string Optional;
// A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.
func (c *Chat) SetUser(user string) {
	c.data.Store("user", user)
}

func (c *Chat) GetHistoryMessages() []map[string]string {
	val, _ := c.data.Load("messages")
	var messages []map[string]string = val.([]map[string]string)
	return messages
}

// NewChat GetOpenAIResponse is the function to get the response from the OpenAI API.
func (c *Chat) NewChat() (*ChatResponse, error) {
	urls := "https://api.openai.com/v1/chat/completions"

	c.mutex.RLock()

	mapVal := map[string]interface{}{}
	c.data.Range(func(key, value interface{}) bool {
		mapVal[key.(string)] = value
		return true
	})

	// convert to json
	jsonBody, err := json.Marshal(mapVal)
	if err != nil {
		return nil, err
	}

	// create request
	req, err := http.NewRequest("POST", urls, bytes.NewBuffer(jsonBody))
	if err != nil {
		return nil, err
	}

	// set authorization key
	key := strings.Builder{}
	key.WriteString("Bearer ")
	key.WriteString(c.key.Load().(string))

	// set headers
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", key.String())

	// send request
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	// read response body
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	res := &ChatResponse{}
	err = json.Unmarshal(body, &res)
	if err != nil {
		return nil, err
	}

	c.mutex.RUnlock()

	if res.Choices == nil {
		return nil, errors.New("no response")
	}

	// Append message of assistant to the messages.
	for index := range res.Choices {
		c.AddMessageAsAssistant(res.Choices[index].Msg.Content)
	}

	return res, nil
}

// NewChatText Get the messages from the response.
func (c *Chat) NewChatText() ([]string, error) {
	res, err := c.NewChat()
	if err != nil {
		return nil, err
	}

	var messages []string
	for index := range res.Choices {
		messages = append(messages, res.Choices[index].Msg.Content)
	}

	return messages, nil
}
