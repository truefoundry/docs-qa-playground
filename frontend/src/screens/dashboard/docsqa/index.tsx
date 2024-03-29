import IconProvider from '@/components/assets/IconProvider'
import Button from '@/components/base/atoms/Button'
import Input from '@/components/base/atoms/Input'
import Markdown from '@/components/base/atoms/Markdown'
import Spinner from '@/components/base/atoms/Spinner/Spinner'
import {
  CollectionQueryDto,
  useGetAllEnabledChatModelsQuery,
  useGetCollectionsQuery,
  useGetOpenapiSpecsQuery,
  useQueryCollectionMutation,
} from '@/stores/qafoundry'
import { MenuItem, Select, TextareaAutosize } from '@mui/material'
import React, { useEffect, useMemo, useState } from 'react'
import NoCollections from './NoCollections'
import SimpleCodeEditor from '@/components/base/molecules/SimpleCodeEditor'
import DocsQaInformation from './DocsQaInformation'

const defaultRetrieverConfig = `{
  "search_type": "similarity",
  "k": 20,
  "fetch_k": 20,
  "filter": {}
}`

const defaultModelConfig = `{
  "parameters": {
    "temperature": 0.1
  }
}`

const defaultPrompt =
  "Given the context, answer the question.\n\nContext: {context}\n'''Question: {question}\nAnswer:"

const DocsQA = () => {
  const [selectedQueryModel, setSelectedQueryModel] = React.useState('')
  const [selectedCollection, setSelectedCollection] = useState('')
  const [selectedRetriever, setSelectedRetriever] = useState('')
  const [prompt, setPrompt] = useState('')
  const [isRunningPrompt, setIsRunningPrompt] = useState(false)
  const [answer, setAnswer] = useState('')
  const [errorMessage, setErrorMessage] = useState(false)
  const [modelConfig, setModelConfig] = useState(defaultModelConfig)
  const [retrieverConfig, setRetrieverConfig] = useState(defaultRetrieverConfig)
  const [promptTemplate, setPromptTemplate] = useState(defaultPrompt)

  const { data: collections, isLoading } = useGetCollectionsQuery()
  const { data: allEnabledModels } = useGetAllEnabledChatModelsQuery()
  const { data: openapiSpecs } = useGetOpenapiSpecsQuery()
  const [searchAnswer] = useQueryCollectionMutation()

  const allRetrieverOptions = useMemo(() => {
    if (!openapiSpecs?.paths) return []
    return Object.keys(openapiSpecs?.paths)
      .filter((path) => path.includes('/retrievers/'))
      .map((str) => str.substring('/retrievers/'.length))
  }, [openapiSpecs])

  const handlePromptSubmit = async () => {
    setIsRunningPrompt(true)
    setAnswer('')
    setErrorMessage(false)
    try {
      const selectedModel = allEnabledModels.find(
        (model: any) => model.id == selectedQueryModel
      )
      if (!selectedModel) {
        throw new Error('Model not found')
      }
      try {
        JSON.parse(modelConfig)
      } catch (err: any) {
        throw new Error('Invalid Model Configuration')
      }
      try {
        JSON.parse(retrieverConfig)
      } catch (err: any) {
        throw new Error('Invalid Retriever Configuration')
      }
      const name = `${selectedModel?.provider_account_name}/${selectedModel?.name}`
      const params: CollectionQueryDto = Object.assign(
        {
          collection_name: selectedCollection,
          query: prompt,
          model_configuration: {
            name: name,
            ...JSON.parse(modelConfig),
          },
          retriever_config: JSON.parse(retrieverConfig),
          prompt_template: promptTemplate,
        },
        {}
      )
      const res: any = await searchAnswer({
        ...params,
        retrieverName: selectedRetriever,
      })
      if (res?.error) {
        setErrorMessage(true)
      } else {
        setAnswer(res.data.answer)
      }
    } catch (err: any) {
      setErrorMessage(true)
    }
    setIsRunningPrompt(false)
  }

  const resetQA = () => {
    setAnswer('')
    setErrorMessage(false)
    setPrompt('')
  }

  useEffect(() => {
    if (allEnabledModels && allEnabledModels.length) {
      setSelectedQueryModel(allEnabledModels[0].id)
    }
  }, [allEnabledModels])

  useEffect(() => {
    if (allRetrieverOptions && allRetrieverOptions.length) {
      setSelectedRetriever(allRetrieverOptions[0])
    }
  }, [allRetrieverOptions])

  useEffect(() => {
    if (collections && collections.length) {
      setSelectedCollection(collections[0].name)
    }
  }, [collections])

  return (
    <>
      <div className="flex gap-5 h-[calc(100vh-104px)] w-full">
        {isLoading ? (
          <div className="h-full w-full flex items-center">
            <Spinner center big />
          </div>
        ) : selectedCollection ? (
          <>
            <div className="h-full border rounded-lg border-[#CEE0F8] w-[380px] bg-white p-4 overflow-auto">
              <div className="flex justify-between items-center mb-1">
                <div className="text-sm">Collection:</div>
                <Select
                  value={selectedCollection}
                  onChange={(e) => {
                    resetQA()
                    setSelectedCollection(e.target.value)
                  }}
                  placeholder="Select Collection..."
                  sx={{
                    background: 'white',
                    height: '32px',
                    width: '13.1875rem',
                    minWidth: '13.1875rem',
                    border: '1px solid #CEE0F8 !important',
                    outline: 'none !important',
                    '& fieldset': {
                      border: 'none !important',
                    },
                  }}
                >
                  {collections?.map((collection: any) => (
                    <MenuItem value={collection.name} key={collection.name}>
                      {collection.name}
                    </MenuItem>
                  ))}
                </Select>
              </div>
              <div className="flex justify-between items-center mb-1 mt-3">
                <div className="text-sm">Query Controller:</div>
                <Select
                  value={selectedRetriever}
                  onChange={(e) => {
                    setSelectedRetriever(e.target.value)
                  }}
                  placeholder="Select Retriever..."
                  sx={{
                    background: 'white',
                    height: '32px',
                    width: '13.1875rem',
                    minWidth: '13.1875rem',
                    border: '1px solid #CEE0F8 !important',
                    outline: 'none !important',
                    '& fieldset': {
                      border: 'none !important',
                    },
                  }}
                >
                  {allRetrieverOptions?.map((retriever: any) => (
                    <MenuItem value={retriever} key={retriever}>
                      {retriever}
                    </MenuItem>
                  ))}
                </Select>
              </div>
              <div className="flex justify-between items-center mb-1 mt-3">
                <div className="text-sm">Model:</div>
                <Select
                  value={selectedQueryModel}
                  onChange={(e) => {
                    setSelectedQueryModel(e.target.value)
                  }}
                  placeholder="Select Model..."
                  sx={{
                    background: 'white',
                    height: '32px',
                    width: '13.1875rem',
                    minWidth: '13.1875rem',
                    border: '1px solid #CEE0F8 !important',
                    outline: 'none !important',
                    '& fieldset': {
                      border: 'none !important',
                    },
                  }}
                >
                  {allEnabledModels?.map((model: any) => (
                    <MenuItem value={model.id} key={model.id}>
                      {model.provider_account_name}/{model.name}
                    </MenuItem>
                  ))}
                </Select>
              </div>
              <div className="mb-1 mt-3 text-sm">Model Configuration:</div>
              <SimpleCodeEditor
                language="json"
                height={130}
                defaultValue={defaultModelConfig}
                onChange={(updatedConfig) =>
                  setModelConfig(updatedConfig ?? '')
                }
              />
              <div className="mb-1 mt-3 text-sm">Retrievers Configuration:</div>
              <SimpleCodeEditor
                language="json"
                height={140}
                defaultValue={defaultRetrieverConfig}
                onChange={(updatedConfig) =>
                  setRetrieverConfig(updatedConfig ?? '')
                }
              />
              <div className="mb-1 mt-3 text-sm">Prompt Template:</div>
              <TextareaAutosize
                className="w-full h-20 bg-[#f0f7ff] border border-[#CEE0F8] rounded-lg p-2 text-sm"
                placeholder="Enter Prompt Template..."
                minRows={3}
                value={promptTemplate}
                onChange={(e) => setPromptTemplate(e.target.value)}
              />
            </div>
            <div className="h-full border rounded-lg border-[#CEE0F8] w-[calc(100%-400px)] bg-white p-4">
              <div className="flex gap-4 items-center">
                <div className="w-full relative">
                  <Input
                    className="w-full h-[44px] text-sm pr-14"
                    placeholder="Ask any question related to this document"
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                  />
                  <Button
                    icon="paper-plane-top"
                    className="btn-sm absolute right-2 top-[6px]"
                    onClick={handlePromptSubmit}
                    loading={isRunningPrompt}
                    disabled={!prompt || !selectedQueryModel}
                  />
                </div>
              </div>
              {answer ? (
                <div className="overflow-y-auto flex gap-4 mt-7 h-[calc(100%-70px)]">
                  <div className="bg-indigo-400 w-6 h-6 rounded-full flex items-center justify-center mt-0.5">
                    <IconProvider icon="message" className="text-white" />
                  </div>
                  <div className="w-full font-inter text-base">
                    <div className="font-bold text-lg">Answer</div>
                    <Markdown>{answer}</Markdown>
                  </div>
                </div>
              ) : isRunningPrompt ? (
                <div className="overflow-y-auto flex flex-col justify-center items-center gap-2 h-[calc(100%-70px)]">
                  <div>
                    <Spinner center medium />
                  </div>
                  <div className="text-center">Fetching Answer...</div>
                </div>
              ) : errorMessage ? (
                <div className="overflow-y-auto flex gap-4 mt-7">
                  <div className="bg-error w-6 h-6 rounded-full flex items-center justify-center mt-0.5">
                    <IconProvider icon="message" className="text-white" />
                  </div>
                  <div className="w-full font-inter text-base text-error">
                    <div className="font-bold text-lg">Error</div>
                    We failed to get answer for your query, please try again by
                    resending query or try again in some time.
                  </div>
                </div>
              ) : (
                <div className="h-[calc(100%-50px)] flex justify-center items-center overflow-y-auto">
                  <div className="min-h-[23rem]">
                    <DocsQaInformation
                      header={'Welcome to DocsQA'}
                      subHeader={
                        <>
                          <p className="text-center max-w-[450px] mt-2">
                            Select a collection from sidebar,
                            <br /> review all the settings and start asking
                            Questions
                          </p>
                        </>
                      }
                    />
                  </div>
                </div>
              )}
            </div>
          </>
        ) : (
          <NoCollections fullWidth />
        )}
      </div>
    </>
  )
}

export default DocsQA
