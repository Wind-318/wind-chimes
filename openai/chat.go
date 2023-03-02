// @file chat.go
// @brief Chat API implementation for OpenAI GPT-3.5 API. (https://beta.openai.com/docs/api-reference/chat)

// Copyright (c) 2023 Wind. All rights reserved.
// Use of this source code is governed by a MIT license
// that can be found in the LICENSE file.

package openai

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"strings"
)

// The message object is used to represent a message in a conversation.
type OpenAIMessage struct {
	// Role is the role of the message. Can be "user" or "system".
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
	// Message is the message object is used to represent a message in a conversation.
	Message OpenAIMessage `json:"message"`
	// FinishReason is the reason the chat completion stopped.
	Finish_reason string `json:"finish_reason"`
}

// ChatCompletion is the chat completion object is used to represent a chat completion.
type ChatResponse struct {
	// ID is the ID of the chat completion.
	Id string `json:"id"`
	// Object is the object type of the chat completion.
	Object string `json:"object"`
	// Created is the timestamp of when the chat completion was created.
	Created int `json:"created"`
	// Model is the ID of the model used to generate the chat completion.
	Choices []Choice `json:"choices"`
	// Usage is the usage object is used to represent the usage of the API.
	Usages Usage `json:"usage"`
}

type Chat struct {
	Data map[string]interface{}
	Key  string
}

// Set Authorization key
func (c *Chat) SetAuthorizationKey(key string) {
	c.Key = key
}

// Add a message to the chat.
// The role can be "user" or "system".
func (c *Chat) AddMessage(role, content string) {
	if _, ok := c.Data["messages"]; !ok {
		c.Data["messages"] = []map[string]string{}
	}

	c.Data["messages"] = append(c.Data["messages"].([]map[string]string), map[string]string{
		"role":    role,
		"content": content,
	})
}

// temperature number Optional Defaults to 1;
// What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random
// while lower values like 0.2 will make it more focused and deterministic.
// We generally recommend altering this or top_p but not both.
func (c *Chat) SetTemperature(temperature float64) {
	c.Data["temperature"] = temperature
}

// top.p number Optional Defaults to 1;
// An alternative to sampling with temperature, called nucleus sampling,
// where the model considers the results of the tokens with top_p probability mass.
// So 0.1 means only the tokens comprising the top 10% probability mass are considered.
// We generally recommend altering this or temperature but not both.
func (c *Chat) SetTopP(top_p float64) {
	c.Data["top_p"] = top_p
}

// How many chat completion choices to generate for each input message.
func (c *Chat) SetN(n int) {
	c.Data["n"] = n
}

// stream boolean Optional Defaults to false.
// If set, partial message deltas will be sent, like in ChatGPT.
// Tokens will be sent as data-only server-sent events as they become available,
// with the stream terminated by a data: [DONE] message.
func (c *Chat) SetStream(stream bool) {
	c.Data["stream"] = stream
}

// stop string or array Optional Defaults to null;
// Up to 4 sequences where the API will stop generating further tokens.
func (c *Chat) SetStopStr(stop string) {
	c.Data["stop"] = stop
}

// stop string or array Optional Defaults to null;
// Up to 4 sequences where the API will stop generating further tokens.
func (c *Chat) SetStopArr(stop []string) {
	c.Data["stop"] = stop
}

// max_tokens integer Optional Defaults to inf;
// The maximum number of tokens allowed for the generated answer.
// By default, the number of tokens the model can return will be (4096 - prompt tokens).
func (c *Chat) SetMaxTokens(max_tokens int) {
	c.Data["max_tokens"] = max_tokens
}

// presence_penalty number Optional Defaults to 0;
// Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear
// in the text so far, increasing the model's likelihood to talk about new topics.
func (c *Chat) SetPresencePenalty(presence_penalty float64) {
	c.Data["presence_penalty"] = presence_penalty
}

// frequency_penalty number Optional Defaults to 0;
// Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing
// frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
func (c *Chat) SetFrequencyPenalty(frequency_penalty float64) {
	c.Data["frequency_penalty"] = frequency_penalty
}

// logit_bias map Optional Defaults to null;
// Modify the likelihood of specified tokens appearing in the completion.
// Accepts a json object that maps tokens (specified by their token ID in the tokenizer)
// to an associated bias value from -100 to 100. Mathematically, the bias is added to the logits
// generated by the model prior to sampling. The exact effect will vary per model, but values
// between -1 and 1 should decrease or increase likelihood of selection; values like -100 or 100
// should result in a ban or exclusive selection of the relevant token.
func (c *Chat) SetLogitBias(logit_bias map[string]int) {
	c.Data["logit_bias"] = logit_bias
}

// user string Optional;
// A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.
func (c *Chat) SetUser(user string) {
	c.Data["user"] = user
}

// GetOpenAIResponse is the function to get the response from the OpenAI API.
func NewChat(c *Chat) (*ChatResponse, error) {
	urls := "https://api.openai.com/v1/chat/completions"

	c.Data["model"] = "gpt-3.5-turbo"

	// convert to json
	jsonBody, err := json.Marshal(c.Data)
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
	key.WriteString(c.Key)

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

	return res, nil
}

// Get the messages from the response.
func NewChatText(c *Chat) ([]string, error) {
	res, err := NewChat(c)
	if err != nil {
		return nil, err
	}

	var messages []string
	for index := range res.Choices {
		messages = append(messages, res.Choices[index].Message.Content)
	}

	return messages, nil
}
