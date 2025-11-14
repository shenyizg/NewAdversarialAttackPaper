# Latest Large Language Model Attack Papers
**update at 2025-11-14 10:15:40**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Sure! Here's a short and concise title for your paper: "Contamination in Generated Text Detection Benchmarks"**

当然！这是您论文的一个简短而简洁的标题：“生成文本检测基准中的污染” cs.LG

published at CSCML 2025

**SubmitDate**: 2025-11-13    [abs](http://arxiv.org/abs/2511.09200v1) [paper-pdf](None)

**Authors**: Philipp Dingfelder, Christian Riess

**Abstract**: Large language models are increasingly used for many applications. To prevent illicit use, it is desirable to be able to detect AI-generated text. Training and evaluation of such detectors critically depend on suitable benchmark datasets. Several groups took on the tedious work of collecting, curating, and publishing large and diverse datasets for this task. However, it remains an open challenge to ensure high quality in all relevant aspects of such a dataset. For example, the DetectRL benchmark exhibits relatively simple patterns of AI-generation in 98.5% of the Claude-LLM data. These patterns may include introductory words such as "Sure! Here is the academic article abstract:", or instances where the LLM rejects the prompted task. In this work, we demonstrate that detectors trained on such data use such patterns as shortcuts, which facilitates spoofing attacks on the trained detectors. We consequently reprocessed the DetectRL dataset with several cleansing operations. Experiments show that such data cleansing makes direct attacks more difficult. The reprocessed dataset is publicly available.

摘要: 大型语言模型越来越多地用于许多应用程序。为了防止非法使用，希望能够检测人工智能生成的文本。此类检测器的培训和评估严重依赖于合适的基准数据集。几个小组承担了为这项任务收集、策划和发布大型多样化数据集的繁琐工作。然而，确保此类数据集所有相关方面的高质量仍然是一个悬而未决的挑战。例如，DetectRL基准测试在98.5%的Claude-LLM数据中显示出相对简单的人工智能生成模式。这些模式可能包括介绍性词语，例如“当然！以下是学术文章摘要：“，或者LLM拒绝提示任务的例子。在这项工作中，我们证明了在此类数据上训练的检测器使用快捷方式等模式，这促进了对训练有素的检测器的欺骗攻击。因此，我们通过几次清理操作重新处理了DetectRL数据集。实验表明，这种数据清理使直接攻击变得更加困难。重新处理的数据集是公开的。



## **2. Uncovering Pretraining Code in LLMs: A Syntax-Aware Attribution Approach**

LLM中预训练代码的发现：一种语法感知的归因方法 cs.CR

Paper has been accepted by AAAI 2026

**SubmitDate**: 2025-11-11    [abs](http://arxiv.org/abs/2511.07033v1) [paper-pdf](None)

**Authors**: Yuanheng Li, Zhuoyang Chen, Xiaoyun Liu, Yuhao Wang, Mingwei Liu, Yang Shi, Kaifeng Huang, Shengjie Zhao

**Abstract**: As large language models (LLMs) become increasingly capable, concerns over the unauthorized use of copyrighted and licensed content in their training data have grown, especially in the context of code. Open-source code, often protected by open source licenses (e.g, GPL), poses legal and ethical challenges when used in pretraining. Detecting whether specific code samples were included in LLM training data is thus critical for transparency, accountability, and copyright compliance. We propose SynPrune, a syntax-pruned membership inference attack method tailored for code. Unlike prior MIA approaches that treat code as plain text, SynPrune leverages the structured and rule-governed nature of programming languages. Specifically, it identifies and excludes consequent tokens that are syntactically required and not reflective of authorship, from attribution when computing membership scores. Experimental results show that SynPrune consistently outperforms the state-of-the-arts. Our method is also robust across varying function lengths and syntax categories.

摘要: 随着大型语言模型（LLM）的能力越来越强，人们对在训练数据中未经授权使用受版权和许可的内容的担忧也越来越大，尤其是在代码环境中。开源代码通常受开源许可证（例如，GPT）保护，在预培训中使用时会带来法律和道德挑战。因此，检测特定代码样本是否包含在LLM训练数据中对于透明度、问责制和版权合规性至关重要。我们提出了SynPrune，这是一种为代码量身定制的语法修剪成员资格推理攻击方法。与之前将代码视为纯文本的MIA方法不同，SynPrune利用了编程语言的结构化和规则管辖性质。具体来说，在计算会员资格分数时，它识别并排除语法上需要且不反映作者身份的后续标记。实验结果表明，SynPrune始终优于最新技术。我们的方法在不同的函数长度和语法类别中也很稳健。



## **3. RAG-targeted Adversarial Attack on LLM-based Threat Detection and Mitigation Framework**

基于LLM的威胁检测和缓解框架的RAG针对性对抗攻击 cs.CR

**SubmitDate**: 2025-11-11    [abs](http://arxiv.org/abs/2511.06212v1) [paper-pdf](None)

**Authors**: Seif Ikbarieh, Kshitiz Aryal, Maanak Gupta

**Abstract**: The rapid expansion of the Internet of Things (IoT) is reshaping communication and operational practices across industries, but it also broadens the attack surface and increases susceptibility to security breaches. Artificial Intelligence has become a valuable solution in securing IoT networks, with Large Language Models (LLMs) enabling automated attack behavior analysis and mitigation suggestion in Network Intrusion Detection Systems (NIDS). Despite advancements, the use of LLMs in such systems further expands the attack surface, putting entire networks at risk by introducing vulnerabilities such as prompt injection and data poisoning. In this work, we attack an LLM-based IoT attack analysis and mitigation framework to test its adversarial robustness. We construct an attack description dataset and use it in a targeted data poisoning attack that applies word-level, meaning-preserving perturbations to corrupt the Retrieval-Augmented Generation (RAG) knowledge base of the framework. We then compare pre-attack and post-attack mitigation responses from the target model, ChatGPT-5 Thinking, to measure the impact of the attack on model performance, using an established evaluation rubric designed for human experts and judge LLMs. Our results show that small perturbations degrade LLM performance by weakening the linkage between observed network traffic features and attack behavior, and by reducing the specificity and practicality of recommended mitigations for resource-constrained devices.

摘要: 物联网（IOT）的快速扩张正在重塑各个行业的通信和运营实践，但它也拓宽了攻击面并增加了安全漏洞的易感性。人工智能已成为保护物联网网络的宝贵解决方案，大型语言模型（LLM）支持网络入侵检测系统（NIDS）中的自动攻击行为分析和缓解建议。尽管取得了进步，但在此类系统中使用LLM进一步扩大了攻击面，通过引入即时注入和数据中毒等漏洞而使整个网络面临风险。在这项工作中，我们攻击了基于LLM的物联网攻击分析和缓解框架，以测试其对抗稳健性。我们构建一个攻击描述数据集，并将其用于有针对性的数据中毒攻击，该攻击应用词级、保留意义的扰动来破坏框架的检索增强生成（RAG）知识库。然后，我们比较目标模型ChatGPT-5 Thinking的攻击前和攻击后缓解响应，以使用为人类专家和判断LLM设计的既定评估指标来衡量攻击对模型性能的影响。我们的结果表明，微小的扰动会削弱观察到的网络流量特征与攻击行为之间的联系，并降低资源受限设备推荐缓解措施的特定性和实用性，从而降低LLM性能。



## **4. Prompt Injection Vulnerability of Consensus Generating Applications in Digital Democracy**

数字民主中共识生成应用程序的即时注入漏洞 cs.CY

27 pages, 16 figures

**SubmitDate**: 2025-11-11    [abs](http://arxiv.org/abs/2508.04281v2) [paper-pdf](None)

**Authors**: Jairo Gudiño-Rosero, Clément Contet, Umberto Grandi, César A. Hidalgo

**Abstract**: Large Language Models (LLMs) are gaining traction as a method to generate consensus statements and aggregate preferences in digital democracy experiments. Yet, LLMs could introduce critical vulnerabilities in these systems. Here, we explore the vulnerability of some off-the-shelf LLMs to prompt-injection attacks in consensus generating systems using a four-dimensional taxonomy of attacks. In LLaMA 3.1 8B and Chat GPT 4.1 Nano, we find LLMs to be more vulnerable to attacks using disagreeable prompts and when targeting situations with unclear consensus. We also find evidence of more effective manipulation when using explicit imperatives and rational-sounding arguments compared to emotional language or fabricated statistics. To mitigate these vulnerabilities, we apply Direct Preference Optimization (DPO), an alignment method that fine-tunes LLMs to prefer unperturbed consensus statements. While DPO and additional layered defenses significantly improve robustness, it still offers limited protection against attacks targeting ambiguous consensus. These results advance our understanding of the vulnerability and robustness of consensus generating LLMs in digital democracy applications.

摘要: 大型语言模型（LLM）作为在数字民主实验中生成共识声明和汇总偏好的方法越来越受欢迎。然而，LLM可能会在这些系统中引入关键漏洞。在这里，我们使用四维攻击分类法探索了一些现成的LLM在共识生成系统中遭受预算注入攻击的脆弱性。在LLaMA 3.1 8B和Chat GPT 4.1 Nano中，我们发现LLM更容易受到使用令人不快的提示以及针对共识不明确的情况的攻击。我们还发现，与情感语言或捏造的统计数据相比，使用明确的命令和听起来合理的论点可以更有效地操纵。为了缓解这些漏洞，我们应用了直接偏好优化（DPO），这是一种对齐方法，可以微调LLM以偏好未受干扰的共识陈述。虽然DPO和额外的分层防御显着提高了稳健性，但它仍然对针对模糊共识的攻击提供有限的保护。这些结果促进了我们对数字民主应用中产生共识的LLM的脆弱性和稳健性的理解。



## **5. LoopLLM: Transferable Energy-Latency Attacks in LLMs via Repetitive Generation**

LoopLLM：通过重复生成对LLM进行可转移能量延迟攻击 cs.CR

14 pages with 7 figures; accepted by the AAAI 2026

**SubmitDate**: 2025-11-12    [abs](http://arxiv.org/abs/2511.07876v1) [paper-pdf](None)

**Authors**: Xingyu Li, Xiaolei Liu, Cheng Liu, Yixiao Xu, Kangyi Ding, Bangzhou Xin, Jia-Li Yin

**Abstract**: As large language models (LLMs) scale, their inference incurs substantial computational resources, exposing them to energy-latency attacks, where crafted prompts induce high energy and latency cost. Existing attack methods aim to prolong output by delaying the generation of termination symbols. However, as the output grows longer, controlling the termination symbols through input becomes difficult, making these methods less effective. Therefore, we propose LoopLLM, an energy-latency attack framework based on the observation that repetitive generation can trigger low-entropy decoding loops, reliably compelling LLMs to generate until their output limits. LoopLLM introduces (1) a repetition-inducing prompt optimization that exploits autoregressive vulnerabilities to induce repetitive generation, and (2) a token-aligned ensemble optimization that aggregates gradients to improve cross-model transferability. Extensive experiments on 12 open-source and 2 commercial LLMs show that LoopLLM significantly outperforms existing methods, achieving over 90% of the maximum output length, compared to 20% for baselines, and improving transferability by around 40% to DeepSeek-V3 and Gemini 2.5 Flash.

摘要: 随着大型语言模型（LLM）的规模化，它们的推断会产生大量的计算资源，使它们面临能量延迟攻击，其中精心设计的提示会导致高能量和延迟成本。现有的攻击方法旨在通过延迟终止符号的生成来延长输出。然而，随着输出的时间越来越长，通过输入控制终止符号变得困难，从而使这些方法变得不那么有效。因此，我们提出了LoopLLM，这是一种能量延迟攻击框架，其基础是重复生成可以触发低熵解码循环的观察，可靠地迫使LLM生成直到其输出限制。LoopLLM引入了（1）诱导重复的即时优化，利用自回归漏洞来诱导重复生成，以及（2）标记对齐的集成优化，聚合梯度以提高跨模型的可移植性。对12个开源LLM和2个商业LLM的广泛实验表明，LoopLLM的性能显着优于现有方法，实现了最大输出长度的90%以上，而基线为20%，并将可移植性提高了约40%至DeepSeek-V3和Gemini 2.5 Flash。



## **6. MSCR: Exploring the Vulnerability of LLMs' Mathematical Reasoning Abilities Using Multi-Source Candidate Replacement**

MSR：使用多源候选替换探索LLM数学推理能力的脆弱性 cs.AI

**SubmitDate**: 2025-11-12    [abs](http://arxiv.org/abs/2511.08055v1) [paper-pdf](None)

**Authors**: Zhishen Sun, Guang Dai, Haishan Ye

**Abstract**: LLMs demonstrate performance comparable to human abilities in complex tasks such as mathematical reasoning, but their robustness in mathematical reasoning under minor input perturbations still lacks systematic investigation. Existing methods generally suffer from limited scalability, weak semantic preservation, and high costs. Therefore, we propose MSCR, an automated adversarial attack method based on multi-source candidate replacement. By combining three information sources including cosine similarity in the embedding space of LLMs, the WordNet dictionary, and contextual predictions from a masked language model, we generate for each word in the input question a set of semantically similar candidates, which are then filtered and substituted one by one to carry out the attack. We conduct large-scale experiments on LLMs using the GSM8K and MATH500 benchmarks. The results show that even a slight perturbation involving only a single word can significantly reduce the accuracy of all models, with the maximum drop reaching 49.89% on GSM8K and 35.40% on MATH500, while preserving the high semantic consistency of the perturbed questions. Further analysis reveals that perturbations not only lead to incorrect outputs but also substantially increase the average response length, which results in more redundant reasoning paths and higher computational resource consumption. These findings highlight the robustness deficiencies and efficiency bottlenecks of current LLMs in mathematical reasoning tasks.

摘要: LLM在数学推理等复杂任务中表现出与人类能力相当的性能，但它们在微小输入扰动下的数学推理稳健性仍然缺乏系统性研究。现有的方法通常存在可扩展性有限、语义保留弱和成本高的问题。因此，我们提出了一种基于多源候选替换的自动对抗攻击方法MSR。通过结合三个信息源，包括LLM嵌入空间中的cos相似性、WordNet词典和来自掩蔽语言模型的上下文预测，我们为输入问题中的每个单词生成一组语义相似的候选项，然后逐个过滤和替换以执行攻击。我们使用GSM 8 K和PATH 500基准对LLM进行大规模实验。结果表明，即使是仅涉及单个单词的轻微扰动也会显着降低所有模型的准确性，在GSM 8K上最大降幅达到49.89%，在PATH 500上最大降幅达到35.40%，同时保持了受扰动问题的高度语义一致性。进一步的分析表明，扰动不仅会导致错误的输出，还会大幅增加平均响应长度，从而导致更多冗余的推理路径和更高的计算资源消耗。这些发现凸显了当前LLM在数学推理任务中的鲁棒性缺陷和效率瓶颈。



## **7. Decoding Latent Attack Surfaces in LLMs: Prompt Injection via HTML in Web Summarization**

解码LLM中的潜在攻击：通过Web摘要中的HTML提示注入 cs.CR

**SubmitDate**: 2025-11-12    [abs](http://arxiv.org/abs/2509.05831v3) [paper-pdf](None)

**Authors**: Ishaan Verma, Arsheya Yadav

**Abstract**: Large Language Models (LLMs) are increasingly integrated into web-based systems for content summarization, yet their susceptibility to prompt injection attacks remains a pressing concern. In this study, we explore how non-visible HTML elements such as <meta>, aria-label, and alt attributes can be exploited to embed adversarial instructions without altering the visible content of a webpage. We introduce a novel dataset comprising 280 static web pages, evenly divided between clean and adversarial injected versions, crafted using diverse HTML-based strategies. These pages are processed through a browser automation pipeline to extract both raw HTML and rendered text, closely mimicking real-world LLM deployment scenarios. We evaluate two state-of-the-art open-source models, Llama 4 Scout (Meta) and Gemma 9B IT (Google), on their ability to summarize this content. Using both lexical (ROUGE-L) and semantic (SBERT cosine similarity) metrics, along with manual annotations, we assess the impact of these covert injections. Our findings reveal that over 29% of injected samples led to noticeable changes in the Llama 4 Scout summaries, while Gemma 9B IT showed a lower, yet non-trivial, success rate of 15%. These results highlight a critical and largely overlooked vulnerability in LLM driven web pipelines, where hidden adversarial content can subtly manipulate model outputs. Our work offers a reproducible framework and benchmark for evaluating HTML-based prompt injection and underscores the urgent need for robust mitigation strategies in LLM applications involving web content.

摘要: 大型语言模型（LLM）越来越多地集成到基于Web的内容摘要系统中，但它们对即时注入攻击的敏感性仍然是一个紧迫的问题。在这项研究中，我们探索了如何利用非可见的HTML元素（例如<meta>、咏叹调标签和alt属性）来嵌入对抗性指令，而不改变网页的可见内容。我们引入了一个由280个静态网页组成的新颖数据集，平均分为干净和对抗注入版本，使用不同的基于HTML的策略制作。这些页面通过浏览器自动化管道进行处理，以提取原始HTML和渲染文本，密切模仿现实世界的LLM部署场景。我们评估了两个最先进的开源模型Llama 4 Scout（Meta）和Gemma 9 B IT（Google）总结此内容的能力。使用词汇（ROUGE-L）和语义（SBERT cos相似性）指标以及手动注释，我们评估这些隐蔽注入的影响。我们的研究结果显示，超过29%的注射样本导致Llama 4 Scout总结发生了显着变化，而Gemma 9 B IT的成功率较低，但并非微不足道，为15%。这些结果凸显了LLM驱动的网络管道中一个关键且在很大程度上被忽视的漏洞，其中隐藏的对抗内容可以巧妙地操纵模型输出。我们的工作为评估基于HTML的即时注入提供了一个可重复的框架和基准，并强调了涉及Web内容的LLM应用程序中对稳健的缓解策略的迫切需要。



## **8. Say It Differently: Linguistic Styles as Jailbreak Vectors**

不同地说：作为越狱载体的语言风格 cs.CL

**SubmitDate**: **NEW** 2025-11-14    [abs](http://arxiv.org/abs/2511.10519v1) [paper-pdf](None)

**Authors**: Srikant Panda, Avinash Rai

**Abstract**: Large Language Models (LLMs) are commonly evaluated for robustness against paraphrased or semantically equivalent jailbreak prompts, yet little attention has been paid to linguistic variation as an attack surface. In this work, we systematically study how linguistic styles such as fear or curiosity can reframe harmful intent and elicit unsafe responses from aligned models. We construct style-augmented jailbreak benchmark by transforming prompts from 3 standard datasets into 11 distinct linguistic styles using handcrafted templates and LLM-based rewrites, while preserving semantic intent. Evaluating 16 open- and close-source instruction-tuned models, we find that stylistic reframing increases jailbreak success rates by up to +57 percentage points. Styles such as fearful, curious and compassionate are most effective and contextualized rewrites outperform templated variants.   To mitigate this, we introduce a style neutralization preprocessing step using a secondary LLM to strip manipulative stylistic cues from user inputs, significantly reducing jailbreak success rates. Our findings reveal a systemic and scaling-resistant vulnerability overlooked in current safety pipelines.

摘要: 大型语言模型（LLM）通常会针对重述或语义等效的越狱提示进行鲁棒性评估，但很少有人关注语言变化作为攻击面。在这项工作中，我们系统地研究恐惧或好奇心等语言风格如何重新定义有害意图并引发一致模型的不安全反应。我们通过使用手工制作的模板和基于LLM的重写将3个标准数据集的提示转换为11种不同的语言风格，同时保留语义意图，来构建风格增强的越狱基准。在评估16个开放和封闭源的描述调整模型后，我们发现风格重组可将越狱成功率提高高达+57个百分点。恐惧、好奇和富有同情心等风格是最有效的，并且背景化的重写优于模板化的变体。   为了缓解这一问题，我们引入了风格中和预处理步骤，使用二级LLM来从用户输入中去除操纵性风格线索，从而显着降低越狱成功率。我们的研究结果揭示了当前安全管道中忽视的系统性和抗扩展性漏洞。



## **9. Speech-Audio Compositional Attacks on Multimodal LLMs and Their Mitigation with SALMONN-Guard**

对多模式LLM的语音音频合成攻击及其使用SALMONN-Guard的缓解 cs.SD

**SubmitDate**: **NEW** 2025-11-14    [abs](http://arxiv.org/abs/2511.10222v1) [paper-pdf](None)

**Authors**: Yudong Yang, Xuezhen Zhang, Zhifeng Han, Siyin Wang, Jimin Zhuang, Zengrui Jin, Jing Shao, Guangzhi Sun, Chao Zhang

**Abstract**: Recent progress in large language models (LLMs) has enabled understanding of both speech and non-speech audio, but exposing new safety risks emerging from complex audio inputs that are inadequately handled by current safeguards. We introduce SACRED-Bench (Speech-Audio Composition for RED-teaming) to evaluate the robustness of LLMs under complex audio-based attacks. Unlike existing perturbation-based methods that rely on noise optimization or white-box access, SACRED-Bench exploits speech-audio composition mechanisms. SACRED-Bench adopts three mechanisms: (a) speech overlap and multi-speaker dialogue, which embeds harmful prompts beneath or alongside benign speech; (b) speech-audio mixture, which imply unsafe intent via non-speech audio alongside benign speech or audio; and (c) diverse spoken instruction formats (open-ended QA, yes/no) that evade text-only filters. Experiments show that, even Gemini 2.5 Pro, the state-of-the-art proprietary LLM, still exhibits 66% attack success rate in SACRED-Bench test set, exposing vulnerabilities under cross-modal, speech-audio composition attacks. To bridge this gap, we propose SALMONN-Guard, a safeguard LLM that jointly inspects speech, audio, and text for safety judgments, reducing attack success down to 20%. Our results highlight the need for audio-aware defenses for the safety of multimodal LLMs. The benchmark and SALMONN-Guard checkpoints can be found at https://huggingface.co/datasets/tsinghua-ee/SACRED-Bench. Warning: this paper includes examples that may be offensive or harmful.

摘要: 大型语言模型（LLM）的最新进展使人们能够理解语音和非语音音频，但也暴露了当前保护措施未充分处理的复杂音频输入所出现的新安全风险。我们引入SACRED-Bench（用于RED团队的语音音频合成）来评估LLM在复杂的基于音频的攻击下的稳健性。与现有的依赖于噪音优化或白盒访问的基于扰动的方法不同，SACRED-Bench利用了语音音频合成机制。SACRED-Bench采用三种机制：（a）语音重叠和多说话者对话，将有害提示嵌入良性语音之下或旁边;（b）语音音频混合，通过非语音音频与良性语音或音频一起暗示不安全意图;（c）规避纯文本过滤器的多种口语指令格式（开放式QA，是/否）。实验表明，即使是最先进的专有LLM Gemini 2.5 Pro，在SACRED-Bench测试集中仍然表现出66%的攻击成功率，暴露了跨模式、语音音频合成攻击下的漏洞。为了弥合这一差距，我们提出SALMONN-Guard，这是一种保护LLM，可联合检查语音、音频和文本以进行安全判断，将攻击成功率降低至20%。我们的结果凸显了为多模式LLM的安全性而需要音频感知防御。基准检查站和SALMONS-Guard检查站可在https://huggingface.co/datasets/tsinghua-ee/SACRED-Bench上找到。警告：本文包含可能令人反感或有害的示例。



## **10. MTAttack: Multi-Target Backdoor Attacks against Large Vision-Language Models**

MTA ttack：针对大型视觉语言模型的多目标后门攻击 cs.CV

AAAI2026, with supplementary material

**SubmitDate**: **NEW** 2025-11-14    [abs](http://arxiv.org/abs/2511.10098v1) [paper-pdf](None)

**Authors**: Zihan Wang, Guansong Pang, Wenjun Miao, Jin Zheng, Xiao Bai

**Abstract**: Recent advances in Large Visual Language Models (LVLMs) have demonstrated impressive performance across various vision-language tasks by leveraging large-scale image-text pretraining and instruction tuning. However, the security vulnerabilities of LVLMs have become increasingly concerning, particularly their susceptibility to backdoor attacks. Existing backdoor attacks focus on single-target attacks, i.e., targeting a single malicious output associated with a specific trigger. In this work, we uncover multi-target backdoor attacks, where multiple independent triggers corresponding to different attack targets are added in a single pass of training, posing a greater threat to LVLMs in real-world applications. Executing such attacks in LVLMs is challenging since there can be many incorrect trigger-target mappings due to severe feature interference among different triggers. To address this challenge, we propose MTAttack, the first multi-target backdoor attack framework for enforcing accurate multiple trigger-target mappings in LVLMs. The core of MTAttack is a novel optimization method with two constraints, namely Proxy Space Partitioning constraint and Trigger Prototype Anchoring constraint. It jointly optimizes multiple triggers in the latent space, with each trigger independently mapping clean images to a unique proxy class while at the same time guaranteeing their separability. Experiments on popular benchmarks demonstrate a high success rate of MTAttack for multi-target attacks, substantially outperforming existing attack methods. Furthermore, our attack exhibits strong generalizability across datasets and robustness against backdoor defense strategies. These findings highlight the vulnerability of LVLMs to multi-target backdoor attacks and underscore the urgent need for mitigating such threats. Code is available at https://github.com/mala-lab/MTAttack.

摘要: 大型视觉语言模型（LVLM）的最新进展通过利用大规模图像-文本预训练和指令调优，在各种视觉语言任务中展示了令人印象深刻的性能。然而，LVLM的安全漏洞变得越来越令人担忧，特别是它们容易受到后门攻击。现有的后门攻击集中在单目标攻击上，即针对与特定触发器关联的单个恶意输出。在这项工作中，我们发现了多目标后门攻击，即在一次训练中添加对应不同攻击目标的多个独立触发器，对现实世界应用中的LVLM构成了更大的威胁。在LVLM中执行此类攻击具有挑战性，因为由于不同触发器之间的严重特征干扰，可能会出现许多不正确的攻击者目标映射。为了应对这一挑战，我们提出了MTA tack，这是第一个多目标后门攻击框架，用于在LVLM中实施准确的多个攻击者-目标映射。MTA ttack的核心是一种具有两个约束的新型优化方法，即代理空间分区约束和触发器原型锚定约束。它联合优化潜在空间中的多个触发器，每个触发器独立地将干净的图像映射到唯一的代理类，同时保证它们的可分离性。流行基准测试的实验表明，MTA ttack对多目标攻击的成功率很高，大大优于现有的攻击方法。此外，我们的攻击在数据集中表现出很强的通用性以及针对后门防御策略的鲁棒性。这些发现凸显了LVLM对多目标后门攻击的脆弱性，并强调了缓解此类威胁的迫切需要。代码可在https://github.com/mala-lab/MTAttack上获取。



## **11. Phantom Menace: Exploring and Enhancing the Robustness of VLA Models against Physical Sensor Attacks**

Phantom Menace：探索和增强VLA模型对物理传感器攻击的鲁棒性 cs.RO

Accepted by AAAI 2026

**SubmitDate**: **NEW** 2025-11-14    [abs](http://arxiv.org/abs/2511.10008v1) [paper-pdf](None)

**Authors**: Xuancun Lu, Jiaxiang Chen, Shilin Xiao, Zizhi Jin, Zhangrui Chen, Hanwen Yu, Bohan Qian, Ruochen Zhou, Xiaoyu Ji, Wenyuan Xu

**Abstract**: Vision-Language-Action (VLA) models revolutionize robotic systems by enabling end-to-end perception-to-action pipelines that integrate multiple sensory modalities, such as visual signals processed by cameras and auditory signals captured by microphones. This multi-modality integration allows VLA models to interpret complex, real-world environments using diverse sensor data streams. Given the fact that VLA-based systems heavily rely on the sensory input, the security of VLA models against physical-world sensor attacks remains critically underexplored.   To address this gap, we present the first systematic study of physical sensor attacks against VLAs, quantifying the influence of sensor attacks and investigating the defenses for VLA models. We introduce a novel ``Real-Sim-Real'' framework that automatically simulates physics-based sensor attack vectors, including six attacks targeting cameras and two targeting microphones, and validates them on real robotic systems. Through large-scale evaluations across various VLA architectures and tasks under varying attack parameters, we demonstrate significant vulnerabilities, with susceptibility patterns that reveal critical dependencies on task types and model designs. We further develop an adversarial-training-based defense that enhances VLA robustness against out-of-distribution physical perturbations caused by sensor attacks while preserving model performance. Our findings expose an urgent need for standardized robustness benchmarks and mitigation strategies to secure VLA deployments in safety-critical environments.

摘要: 视觉-语言-动作（VLA）模型通过实现端到端的感知到动作管道，彻底改变了机器人系统，该管道集成了多种感官模式，例如由摄像机处理的视觉信号和由麦克风捕获的听觉信号。这种多模式集成使VLA模型能够使用不同的传感器数据流来解释复杂的现实世界环境。鉴于基于VLA的系统严重依赖感官输入，VLA模型对抗物理世界传感器攻击的安全性仍然严重不足。   为了弥补这一差距，我们首次对针对VLA的物理传感器攻击进行了系统研究，量化了传感器攻击的影响并调查VLA模型的防御。我们引入了一种新型的“Real-Sim-Real”框架，该框架自动模拟基于物理的传感器攻击载体，包括六次针对摄像头和两个针对麦克风的攻击，并在真实的机器人系统上对其进行验证。通过在不同攻击参数下对各种VLA架构和任务进行大规模评估，我们展示了显着的漏洞，其易感性模式揭示了对任务类型和模型设计的关键依赖性。我们进一步开发了一种基于对抗训练的防御，可以增强VLA对传感器攻击引起的分布外物理扰动的鲁棒性，同时保持模型性能。我们的研究结果揭示了迫切需要标准化的稳健性基准和缓解策略，以确保VLA在安全关键环境中的部署。



## **12. EnchTable: Unified Safety Alignment Transfer in Fine-tuned Large Language Models**

EnchTable：微调大型语言模型中的统一安全对齐转移 cs.CL

Accepted by IEEE Symposium on Security and Privacy (S&P) 2026

**SubmitDate**: **NEW** 2025-11-14    [abs](http://arxiv.org/abs/2511.09880v1) [paper-pdf](None)

**Authors**: Jialin Wu, Kecen Li, Zhicong Huang, Xinfeng Li, Xiaofeng Wang, Cheng Hong

**Abstract**: Many machine learning models are fine-tuned from large language models (LLMs) to achieve high performance in specialized domains like code generation, biomedical analysis, and mathematical problem solving. However, this fine-tuning process often introduces a critical vulnerability: the systematic degradation of safety alignment, undermining ethical guidelines and increasing the risk of harmful outputs. Addressing this challenge, we introduce EnchTable, a novel framework designed to transfer and maintain safety alignment in downstream LLMs without requiring extensive retraining. EnchTable leverages a Neural Tangent Kernel (NTK)-based safety vector distillation method to decouple safety constraints from task-specific reasoning, ensuring compatibility across diverse model architectures and sizes. Additionally, our interference-aware merging technique effectively balances safety and utility, minimizing performance compromises across various task domains. We implemented a fully functional prototype of EnchTable on three different task domains and three distinct LLM architectures, and evaluated its performance through extensive experiments on eleven diverse datasets, assessing both utility and model safety. Our evaluations include LLMs from different vendors, demonstrating EnchTable's generalization capability. Furthermore, EnchTable exhibits robust resistance to static and dynamic jailbreaking attacks, outperforming vendor-released safety models in mitigating adversarial prompts. Comparative analyses with six parameter modification methods and two inference-time alignment baselines reveal that EnchTable achieves a significantly lower unsafe rate, higher utility score, and universal applicability across different task domains. Additionally, we validate EnchTable can be seamlessly integrated into various deployment pipelines without significant overhead.

摘要: 许多机器学习模型都是从大型语言模型（LLM）进行微调的，以在代码生成、生物医学分析和数学问题解决等专业领域实现高性能。然而，这种微调过程往往会引入一个关键的漏洞：安全一致性的系统性退化，破坏道德准则并增加有害输出的风险。为了应对这一挑战，我们引入了EnchTable，这是一个新颖的框架，旨在转移和维护下游LLM的安全一致，而无需进行广泛的再培训。EnchTable利用基于神经切向核（NTK）的安全向量提炼方法将安全约束与特定任务推理脱钩，确保不同模型架构和大小之间的兼容性。此外，我们的干扰感知合并技术有效地平衡了安全性和实用性，最大限度地减少了各个任务域的性能损害。我们在三个不同的任务域和三个不同的LLM架构上实现了功能齐全的EnchTable原型，并通过对十一个不同数据集的广泛实验评估了其性能，评估了效用和模型安全性。我们的评估包括来自不同供应商的LLM，展示了EnchTable的概括能力。此外，EnchTable对静态和动态越狱攻击表现出强大的抵抗力，在减轻对抗提示方面优于供应商发布的安全模型。对六种参数修改方法和两种推断时间对齐基线的比较分析表明，EnchTable实现了显着较低的不安全率、较高的效用评分以及跨不同任务领域的普遍适用性。此外，我们还验证了EnchTable可以无缝集成到各种部署管道中，而无需承担重大费用。



## **13. Adaptive and Robust Data Poisoning Detection and Sanitization in Wearable IoT Systems using Large Language Models**

使用大型语言模型在可穿戴物联网系统中进行自适应和稳健的数据中毒检测和清理 cs.LG

**SubmitDate**: 2025-11-11    [abs](http://arxiv.org/abs/2511.02894v2) [paper-pdf](None)

**Authors**: W. K. M Mithsara, Ning Yang, Ahmed Imteaj, Hussein Zangoti, Abdur R. Shahid

**Abstract**: The widespread integration of wearable sensing devices in Internet of Things (IoT) ecosystems, particularly in healthcare, smart homes, and industrial applications, has required robust human activity recognition (HAR) techniques to improve functionality and user experience. Although machine learning models have advanced HAR, they are increasingly susceptible to data poisoning attacks that compromise the data integrity and reliability of these systems. Conventional approaches to defending against such attacks often require extensive task-specific training with large, labeled datasets, which limits adaptability in dynamic IoT environments. This work proposes a novel framework that uses large language models (LLMs) to perform poisoning detection and sanitization in HAR systems, utilizing zero-shot, one-shot, and few-shot learning paradigms. Our approach incorporates \textit{role play} prompting, whereby the LLM assumes the role of expert to contextualize and evaluate sensor anomalies, and \textit{think step-by-step} reasoning, guiding the LLM to infer poisoning indicators in the raw sensor data and plausible clean alternatives. These strategies minimize reliance on curation of extensive datasets and enable robust, adaptable defense mechanisms in real-time. We perform an extensive evaluation of the framework, quantifying detection accuracy, sanitization quality, latency, and communication cost, thus demonstrating the practicality and effectiveness of LLMs in improving the security and reliability of wearable IoT systems.

摘要: 可穿戴传感设备在物联网（IoT）生态系统中的广泛集成，特别是在医疗保健、智能家居和工业应用中，需要强大的人类活动识别（HAR）技术来改善功能和用户体验。尽管机器学习模型具有高级HAR，但它们越来越容易受到数据中毒攻击，从而损害这些系统的数据完整性和可靠性。防御此类攻击的传统方法通常需要使用大型标记数据集进行广泛的任务特定训练，这限制了动态物联网环境中的适应性。这项工作提出了一种新颖的框架，该框架使用大型语言模型（LLM）在HAR系统中执行中毒检测和清理，利用零触发、单触发和少触发学习范式。我们的方法结合了\textit{role play}提示，LLM承担专家的角色来情境化和评估传感器异常，以及\textit{think分步}推理，指导LLM推断原始传感器数据中的中毒指标和合理的清洁替代品。这些策略最大限度地减少了对大量数据集管理的依赖，并实时实现强大、适应性强的防御机制。我们对框架进行了广泛的评估，量化检测准确性、消毒质量、延迟和通信成本，从而证明了LLM在提高可穿戴物联网系统安全性和可靠性方面的实用性和有效性。



## **14. E2E-VGuard: Adversarial Prevention for Production LLM-based End-To-End Speech Synthesis**

E2 E-VGuard：基于生产LLM的端到端语音合成的对抗预防 cs.SD

Accepted to NeurIPS 2025

**SubmitDate**: 2025-11-11    [abs](http://arxiv.org/abs/2511.07099v1) [paper-pdf](None)

**Authors**: Zhisheng Zhang, Derui Wang, Yifan Mi, Zhiyong Wu, Jie Gao, Yuxin Cao, Kai Ye, Minhui Xue, Jie Hao

**Abstract**: Recent advancements in speech synthesis technology have enriched our daily lives, with high-quality and human-like audio widely adopted across real-world applications. However, malicious exploitation like voice-cloning fraud poses severe security risks. Existing defense techniques struggle to address the production large language model (LLM)-based speech synthesis. While previous studies have considered the protection for fine-tuning synthesizers, they assume manually annotated transcripts. Given the labor intensity of manual annotation, end-to-end (E2E) systems leveraging automatic speech recognition (ASR) to generate transcripts are becoming increasingly prevalent, e.g., voice cloning via commercial APIs. Therefore, this E2E speech synthesis also requires new security mechanisms. To tackle these challenges, we propose E2E-VGuard, a proactive defense framework for two emerging threats: (1) production LLM-based speech synthesis, and (2) the novel attack arising from ASR-driven E2E scenarios. Specifically, we employ the encoder ensemble with a feature extractor to protect timbre, while ASR-targeted adversarial examples disrupt pronunciation. Moreover, we incorporate the psychoacoustic model to ensure perturbative imperceptibility. For a comprehensive evaluation, we test 16 open-source synthesizers and 3 commercial APIs across Chinese and English datasets, confirming E2E-VGuard's effectiveness in timbre and pronunciation protection. Real-world deployment validation is also conducted. Our code and demo page are available at https://wxzyd123.github.io/e2e-vguard/.

摘要: 语音合成技术的最新进步丰富了我们的日常生活，高质量的类人音频在现实世界的应用中广泛采用。然而，语音克隆欺诈等恶意利用会带来严重的安全风险。现有的防御技术难以解决基于生产大语言模型（LLM）的语音合成问题。虽然之前的研究考虑了对微调合成器的保护，但他们假设手动注释的文字记录。鉴于手动注释的劳动强度，利用自动语音识别（ASB）来生成文字记录的端到端（E2 E）系统变得越来越普遍，例如，通过商业API进行语音克隆。因此，这种E2 E语音合成还需要新的安全机制。为了应对这些挑战，我们提出了E2 E-VGuard，这是一个针对两种新兴威胁的主动防御框架：（1）基于LLM的生产语音合成，以及（2）由SVR驱动的E2 E场景引起的新型攻击。具体来说，我们使用带有特征提取器的编码器集成来保护音色，而针对ASB的对抗性示例则会扰乱发音。此外，我们结合了心理声学模型来确保扰动的不可感知性。为了进行全面评估，我们测试了中文和英文数据集的16个开源合成器和3个商业API，确认了E2 E-VGuard在音色和发音保护方面的有效性。还进行现实世界的部署验证。我们的代码和演示页面可访问https://wxzyd123.github.io/e2e-vguard/。



## **15. From Pretrain to Pain: Adversarial Vulnerability of Video Foundation Models Without Task Knowledge**

从预训练到痛苦：没有任务知识的视频基础模型的对抗脆弱性 cs.CV

AAAI 2026 (Oral presentation)

**SubmitDate**: 2025-11-11    [abs](http://arxiv.org/abs/2511.07049v1) [paper-pdf](None)

**Authors**: Hui Lu, Yi Yu, Song Xia, Yiming Yang, Deepu Rajan, Boon Poh Ng, Alex Kot, Xudong Jiang

**Abstract**: Large-scale Video Foundation Models (VFMs) has significantly advanced various video-related tasks, either through task-specific models or Multi-modal Large Language Models (MLLMs). However, the open accessibility of VFMs also introduces critical security risks, as adversaries can exploit full knowledge of the VFMs to launch potent attacks. This paper investigates a novel and practical adversarial threat scenario: attacking downstream models or MLLMs fine-tuned from open-source VFMs, without requiring access to the victim task, training data, model query, and architecture. In contrast to conventional transfer-based attacks that rely on task-aligned surrogate models, we demonstrate that adversarial vulnerabilities can be exploited directly from the VFMs. To this end, we propose the Transferable Video Attack (TVA), a temporal-aware adversarial attack method that leverages the temporal representation dynamics of VFMs to craft effective perturbations. TVA integrates a bidirectional contrastive learning mechanism to maximize the discrepancy between the clean and adversarial features, and introduces a temporal consistency loss that exploits motion cues to enhance the sequential impact of perturbations. TVA avoids the need to train expensive surrogate models or access to domain-specific data, thereby offering a more practical and efficient attack strategy. Extensive experiments across 24 video-related tasks demonstrate the efficacy of TVA against downstream models and MLLMs, revealing a previously underexplored security vulnerability in the deployment of video models.

摘要: 大规模视频基础模型（VFM）通过特定任务的模型或多模式大型语言模型（MLLM）显着推进了各种视频相关任务。然而，VFM的开放访问性也带来了严重的安全风险，因为对手可以利用对VFM的全面了解来发起强有力的攻击。本文研究了一种新颖且实用的对抗性威胁场景：攻击下游模型或从开源VFM微调的MLLM，而不需要访问受害者任务、训练数据、模型查询和架构。与依赖任务对齐代理模型的传统基于传输的攻击相反，我们证明可以直接从VFM利用对抗漏洞。为此，我们提出了可传输视频攻击（TVA），这是一种时间感知的对抗攻击方法，它利用VFM的时间表示动态来制造有效的扰动。TVA集成了双向对比学习机制，以最大化清晰特征和对抗特征之间的差异，并引入了时间一致性损失，利用运动线索来增强扰动的顺序影响。TVA避免了训练昂贵的代理模型或访问特定于领域的数据的需要，从而提供了更实用、更高效的攻击策略。针对24个视频相关任务的广泛实验证明了TVA针对下游模型和MLLM的有效性，揭示了视频模型部署中之前未充分探索的安全漏洞。



## **16. Graph Representation-based Model Poisoning on the Heterogeneous Internet of Agents**

基于图表示的模型中毒在异类代理互联网上 cs.NI

6 pages, 6 figures

**SubmitDate**: 2025-11-11    [abs](http://arxiv.org/abs/2511.07176v1) [paper-pdf](None)

**Authors**: Hanlin Cai, Houtianfu Wang, Haofan Dong, Kai Li, Ozgur B. Akan

**Abstract**: Internet of Agents (IoA) envisions a unified, agent-centric paradigm where heterogeneous large language model (LLM) agents can interconnect and collaborate at scale. Within this paradigm, federated learning (FL) serves as a key enabler that allows distributed LLM agents to co-train global models without centralizing data. However, the FL-enabled IoA system remains vulnerable to model poisoning attacks, and the prevailing distance and similarity-based defenses become fragile at billion-parameter scale and under heterogeneous data distributions. This paper proposes a graph representation-based model poisoning (GRMP) attack, which passively exploits observed benign local models to construct a parameter correlation graph and extends an adversarial variational graph autoencoder to capture and reshape higher-order dependencies. The GRMP attack synthesizes malicious local models that preserve benign-like statistics while embedding adversarial objectives, remaining elusive to detection at the server. Experiments demonstrate a gradual drop in system accuracy under the proposed attack and the ineffectiveness of the prevailing defense mechanism in detecting the attack, underscoring a severe threat to the ambitious IoA paradigm.

摘要: 代理互联网（IoA）设想了一个统一的、以代理为中心的范式，其中异类大型语言模型（LLM）代理可以大规模互连和协作。在此范式中，联合学习（FL）是一个关键推动因素，允许分布式LLM代理在不集中数据的情况下共同训练全球模型。然而，支持FL的IoA系统仍然容易受到模型中毒攻击，并且普遍的基于距离和相似性的防御在十亿参数规模和异类数据分布下变得脆弱。本文提出了一种基于图表示的模型中毒（GRMP）攻击，该攻击被动地利用观察到的良性局部模型来构建参数相关图，并扩展对抗变分图自动编码器来捕获和重塑更高级依赖关系。GRMP攻击合成了恶意的本地模型，这些模型保留了类似善意的统计数据，同时嵌入了敌对目标，但在服务器上仍然难以检测。实验表明，在拟议的攻击下，系统准确性逐渐下降，并且流行的防御机制在检测攻击方面无效，这凸显了雄心勃勃的IoA范式面临的严重威胁。



## **17. DP-Fusion: Token-Level Differentially Private Inference for Large Language Models**

DP-Fusion：大型语言模型的令牌级差异私人推理 cs.CL

Our code and data are publicly available here: https://github.com/MBZUAI-Trustworthy-ML/DP-Fusion-DPI

**SubmitDate**: 2025-11-11    [abs](http://arxiv.org/abs/2507.04531v3) [paper-pdf](None)

**Authors**: Rushil Thareja, Preslav Nakov, Praneeth Vepakomma, Nils Lukas

**Abstract**: Large language models (LLMs) do not preserve privacy at inference-time. The LLM's outputs can inadvertently reveal information about the model's context, which presents a privacy challenge when the LLM is augmented via tools or databases containing sensitive information. Existing privacy-preserving methods at inference-time have significant limitations since they (i) lack provable guarantees or (ii) have a poor utility/privacy trade-off. We propose DP-Fusion, a Differentially Private Inference (DPI) mechanism for LLMs that provably bounds the influence a set of tokens in the context can have on the LLM's output. DP-Fusion works as follows: (1) label a subset of sensitive tokens, (2) infer the LLM without any sensitive tokens to obtain a baseline, (3) infer the LLM with the sensitive tokens, and (4) blend distributions so that the final output remains within a bounded distance of the baseline distribution. While this per-token influence bound also mitigates jailbreak-style prompt injection, we focus on \emph{document privatization}, where the goal is to paraphrase a document containing sensitive tokens, e.g., personally identifiable information, so that no attacker can reliably infer them from the paraphrased document while preserving high text quality. The privacy/utility trade-off is controlled by $ε$, where $ε=0$ hides sensitive tokens entirely, while higher values trade off privacy for improved text quality. We show that our method creates token-level provably privatized documents with substantially improved theoretical and empirical privacy, achieving $6\times$ lower perplexity than related DPI methods.

摘要: 大型语言模型（LLM）在推理时不保护隐私。LLM的输出可能会无意中泄露有关模型上下文的信息，当LLM通过包含敏感信息的工具或数据库进行增强时，这会带来隐私挑战。现有的隐私保护方法在推理时间有显着的局限性，因为它们（i）缺乏可证明的保证或（ii）有一个穷人的效用/隐私权衡。我们提出了DP融合，一个差分私人推理（DPI）机制的LLM，可证明的范围内的一组令牌的上下文中可以对LLM的输出的影响。DP-Fusion的工作原理如下：（1）标记敏感标记的子集，（2）推断没有任何敏感标记的LLM以获得基线，（3）推断具有敏感标记的LLM，以及（4）混合分布，使得最终输出保持在基线分布的有界距离内。虽然这种每个令牌的影响范围也减轻了越狱风格的提示注入，但我们专注于\ldblquote文档私有化“，其目标是解释包含敏感令牌的文档，例如，个人可识别信息，这样攻击者就无法从改述的文档中可靠地推断出它们，同时保持高文本质量。隐私/实用性权衡由$e $控制，其中$e =0$完全隐藏敏感令牌，而更高的值则牺牲隐私以提高文本质量。我们表明，我们的方法创建了代币级的可证明私有化文档，具有显着改善的理论和经验隐私，比相关DPA方法实现了6倍的困惑度。



## **18. AutoAdv: Automated Adversarial Prompting for Multi-Turn Jailbreaking of Large Language Models**

AutoAdv：大型语言模型多回合越狱的自动对抗预算 cs.CL

Accepted to NeurIPS 2025 Lock-LLM Workshop. Code is available at https://github.com/AAN-AutoAdv/AutoAdv

**SubmitDate**: 2025-11-11    [abs](http://arxiv.org/abs/2511.02376v2) [paper-pdf](None)

**Authors**: Aashray Reddy, Andrew Zagula, Nicholas Saban, Kevin Zhu

**Abstract**: Large Language Models (LLMs) remain vulnerable to jailbreaking attacks where adversarial prompts elicit harmful outputs, yet most evaluations focus on single-turn interactions while real-world attacks unfold through adaptive multi-turn conversations. We present AutoAdv, a training-free framework for automated multi-turn jailbreaking that achieves up to 95% attack success rate on Llama-3.1-8B within six turns a 24 percent improvement over single turn baselines. AutoAdv uniquely combines three adaptive mechanisms: a pattern manager that learns from successful attacks to enhance future prompts, a temperature manager that dynamically adjusts sampling parameters based on failure modes, and a two-phase rewriting strategy that disguises harmful requests then iteratively refines them. Extensive evaluation across commercial and open-source models (GPT-4o-mini, Qwen3-235B, Mistral-7B) reveals persistent vulnerabilities in current safety mechanisms, with multi-turn attacks consistently outperforming single-turn approaches. These findings demonstrate that alignment strategies optimized for single-turn interactions fail to maintain robustness across extended conversations, highlighting an urgent need for multi-turn-aware defenses.

摘要: 大型语言模型（LLM）仍然容易受到越狱攻击，其中对抗性提示会引发有害输出，但大多数评估都集中在单轮交互上，而现实世界的攻击则通过自适应多轮对话展开。我们介绍了AutoAdv，这是一个用于自动多回合越狱的免训练框架，在六个回合内对Llama-3.1-8B的攻击成功率高达95%，比单回合基线提高了24%。AutoAdv独特地结合了三种自适应机制：从成功的攻击中学习以增强未来提示的模式管理器、根据失败模式动态调整采样参数的温度管理器以及掩盖有害请求然后迭代细化它们的两阶段重写策略。对商业和开源模型（GPT-4 o-mini、Qwen 3 - 235 B、Mistral-7 B）的广泛评估揭示了当前安全机制中存在的持续漏洞，多回合攻击的表现始终优于单回合方法。这些发现表明，针对单轮交互优化的对齐策略无法在扩展对话中保持稳健性，凸显了对多轮感知防御的迫切需求。



## **19. When AI Meets the Web: Prompt Injection Risks in Third-Party AI Chatbot Plugins**

当AI遇到Web：第三方AI聊天机器人插件中的提示注入风险 cs.CR

At IEEE S&P 2026

**SubmitDate**: 2025-11-11    [abs](http://arxiv.org/abs/2511.05797v1) [paper-pdf](None)

**Authors**: Yigitcan Kaya, Anton Landerer, Stijn Pletinckx, Michelle Zimmermann, Christopher Kruegel, Giovanni Vigna

**Abstract**: Prompt injection attacks pose a critical threat to large language models (LLMs), with prior work focusing on cutting-edge LLM applications like personal copilots. In contrast, simpler LLM applications, such as customer service chatbots, are widespread on the web, yet their security posture and exposure to such attacks remain poorly understood. These applications often rely on third-party chatbot plugins that act as intermediaries to commercial LLM APIs, offering non-expert website builders intuitive ways to customize chatbot behaviors. To bridge this gap, we present the first large-scale study of 17 third-party chatbot plugins used by over 10,000 public websites, uncovering previously unknown prompt injection risks in practice. First, 8 of these plugins (used by 8,000 websites) fail to enforce the integrity of the conversation history transmitted in network requests between the website visitor and the chatbot. This oversight amplifies the impact of direct prompt injection attacks by allowing adversaries to forge conversation histories (including fake system messages), boosting their ability to elicit unintended behavior (e.g., code generation) by 3 to 8x. Second, 15 plugins offer tools, such as web-scraping, to enrich the chatbot's context with website-specific content. However, these tools do not distinguish the website's trusted content (e.g., product descriptions) from untrusted, third-party content (e.g., customer reviews), introducing a risk of indirect prompt injection. Notably, we found that ~13% of e-commerce websites have already exposed their chatbots to third-party content. We systematically evaluate both vulnerabilities through controlled experiments grounded in real-world observations, focusing on factors such as system prompt design and the underlying LLM. Our findings show that many plugins adopt insecure practices that undermine the built-in LLM safeguards.

摘要: 即时注入攻击对大型语言模型（LLM）构成了严重威胁，之前的工作重点是个人并行驾驶等尖端LLM应用程序。相比之下，更简单的LLM应用程序（例如客户服务聊天机器人）在网络上很广泛，但人们对它们的安全姿态和遭受此类攻击的风险仍然知之甚少。这些应用程序通常依赖于第三方聊天机器人插件，这些插件充当商业LLM API的中介，为非专家网站构建者提供自定义聊天机器人行为的直观方法。为了弥合这一差距，我们对10，000多个公共网站使用的17个第三方聊天机器人插件进行了首次大规模研究，揭示了实践中之前未知的即时注入风险。首先，这些插件中有8个（被8,000个网站使用）无法强制执行网站访问者和聊天机器人之间在网络请求中传输的会话历史的完整性。这种疏忽放大了直接提示注入攻击的影响，允许对手伪造会话历史（包括伪造的系统消息），提高了他们引发意外行为的能力（例如，代码生成）3到8倍。其次，15个插件提供了一些工具，比如网页抓取，可以用特定于网站的内容来丰富聊天机器人的上下文。然而，这些工具不能区分网站的可信内容（例如，产品描述）从不可信的第三方内容（例如，客户评论），从而引入间接即时注入的风险。值得注意的是，我们发现约13%的电子商务网站已经将其聊天机器人暴露给第三方内容。我们通过基于现实世界观察的受控实验系统地评估这两个漏洞，重点关注系统提示设计和底层LLM等因素。我们的调查结果表明，许多插件采用不安全的做法，从而破坏了内置的LLM保障措施。



## **20. A Self-Improving Architecture for Dynamic Safety in Large Language Models**

大型语言模型中动态安全的自我改进架构 cs.SE

Under review at the journal Information and Software Technology (Special Issue on Software Architecture for AI-Driven Systems)

**SubmitDate**: 2025-11-12    [abs](http://arxiv.org/abs/2511.07645v1) [paper-pdf](None)

**Authors**: Tyler Slater

**Abstract**: Context: The integration of Large Language Models (LLMs) into core software systems is accelerating. However, existing software architecture patterns are static, while current safety assurance methods are not scalable, leaving systems vulnerable to novel adversarial threats.   Objective: To design, implement, and evaluate a novel software architecture that enables an AI-driven system to autonomously and continuously adapt its own safety protocols at runtime.   Method: We propose the Self-Improving Safety Framework (SISF), a runtime architecture that couples an unprotected, unaligned base LLM (mistralai/Mistral-7B-v0.1) with a dynamic feedback loop. This loop consists of an AI Adjudicator (GPT-4o) for breach detection and a Policy Synthesis Module (GPT-4 Turbo) that autonomously generates new, generalized safety policies (both heuristic and semantic) in response to failures.   Results: We conducted a dynamic learning evaluation using the 520-prompt AdvBench dataset. The unprotected model was 100% vulnerable. Our SISF, starting from zero policies, demonstrated a clear learning curve: it detected 237 breaches, autonomously synthesized 234 new policies, and reduced the overall Attack Success Rate (ASR) to 45.58%. In a subsequent test on 520 benign prompts, the SISF achieved a 0.00% False Positive Rate (FPR), proving its ability to adapt without compromising user utility.   Conclusion: An architectural approach to AI safety, based on the principles of self-adaptation, is a viable and effective strategy. Our framework demonstrates a practical path towards building more robust, resilient, and scalable AI-driven systems, shifting safety assurance from a static, pre-deployment activity to an automated, runtime process.

摘要: 背景：大型语言模型（LLM）与核心软件系统的集成正在加速。然而，现有的软件架构模式是静态的，而当前的安全保证方法不可扩展，导致系统容易受到新型对抗威胁的影响。   目标：设计、实施和评估一种新型软件架构，使人工智能驱动的系统能够在运行时自主、持续地调整自己的安全协议。   方法：我们提出了自我改进安全框架（SISF），这是一种运行时架构，将未受保护的、未对齐的基本LLM（mistralai/Mistral-7B-v0.1）与动态反馈循环相结合。该循环由用于漏洞检测的AI裁决器（GPT-4o）和策略合成模块（GPT-4 Turbo）组成，该模块自主生成新的广义安全策略（启发式和语义）以响应故障。   结果：我们使用520次提示AdvBench数据集进行了动态学习评估。无保护的模型100%容易受到攻击。我们的SISF从零策略开始，展示了清晰的学习曲线：它检测到237个漏洞，自主合成了234个新策略，并将总体攻击成功率（ASB）降低至45.58%。在随后对520个良性提示进行的测试中，SISF实现了0.00%的假阳性率（FPR），证明了其在不损害用户实用性的情况下进行调整的能力。   结论：基于自适应原则的人工智能安全架构方法是一种可行且有效的策略。我们的框架展示了一条构建更强大、更有弹性和可扩展的人工智能驱动系统的实用路径，将安全保证从静态的部署前活动转变为自动化的运行时流程。



## **21. Comparing Reconstruction Attacks on Pretrained Versus Full Fine-tuned Large Language Model Embeddings on Homo Sapiens Splice Sites Genomic Data**

比较对预训练的重建攻击与嵌入智人拼接位点基因组数据的完全微调大语言模型的重建攻击 cs.LG

**SubmitDate**: 2025-11-12    [abs](http://arxiv.org/abs/2511.07481v1) [paper-pdf](None)

**Authors**: Reem Al-Saidi, Erman Ayday, Ziad Kobti

**Abstract**: This study investigates embedding reconstruction attacks in large language models (LLMs) applied to genomic sequences, with a specific focus on how fine-tuning affects vulnerability to these attacks. Building upon Pan et al.'s seminal work demonstrating that embeddings from pretrained language models can leak sensitive information, we conduct a comprehensive analysis using the HS3D genomic dataset to determine whether task-specific optimization strengthens or weakens privacy protections. Our research extends Pan et al.'s work in three significant dimensions. First, we apply their reconstruction attack pipeline to pretrained and fine-tuned model embeddings, addressing a critical gap in their methodology that did not specify embedding types. Second, we implement specialized tokenization mechanisms tailored specifically for DNA sequences, enhancing the model's ability to process genomic data, as these models are pretrained on natural language and not DNA. Third, we perform a detailed comparative analysis examining position-specific, nucleotide-type, and privacy changes between pretrained and fine-tuned embeddings. We assess embeddings vulnerabilities across different types and dimensions, providing deeper insights into how task adaptation shifts privacy risks throughout genomic sequences. Our findings show a clear distinction in reconstruction vulnerability between pretrained and fine-tuned embeddings. Notably, fine-tuning strengthens resistance to reconstruction attacks in multiple architectures -- XLNet (+19.8\%), GPT-2 (+9.8\%), and BERT (+7.8\%) -- pointing to task-specific optimization as a potential privacy enhancement mechanism. These results highlight the need for advanced protective mechanisms for language models processing sensitive genomic data, while highlighting fine-tuning as a potential privacy-enhancing technique worth further exploration.

摘要: 这项研究调查了应用于基因组序列的大型语言模型（LLM）中的嵌入重建攻击，特别关注微调如何影响这些攻击的脆弱性。以潘等人为基础。我们的开创性工作表明，来自预训练语言模型的嵌入可能会泄露敏感信息，我们使用HS 3D基因组数据集进行了全面分析，以确定特定任务的优化是否会加强或削弱隐私保护。我们的研究扩展了Pan等人的工作在三个重要方面。首先，我们将他们的重建攻击管道应用于预训练和微调的模型嵌入，解决了他们方法论中未指定嵌入类型的关键差距。其次，我们实施专门为DNA序列量身定制的专门代币化机制，增强模型处理基因组数据的能力，因为这些模型是在自然语言而不是DNA上预先训练的。第三，我们进行详细的比较分析，检查预训练嵌入和微调嵌入之间的位置特定、核苷类型和隐私变化。我们评估不同类型和维度的嵌入漏洞，为任务适应如何在整个基因组序列中转移隐私风险提供更深入的见解。我们的研究结果表明，预训练嵌入和微调嵌入之间的重建脆弱性存在明显差异。值得注意的是，微调增强了对多种架构（XLNet（+19.8%%）、GPT-2（+9.8%%）和BERT（+7.8%%））中重建攻击的抵抗力，指出特定任务的优化是一种潜在的隐私增强机制。这些结果凸显了处理敏感基因组数据的语言模型需要高级保护机制，同时强调微调是一种值得进一步探索的潜在隐私增强技术。



## **22. Reasoning Up the Instruction Ladder for Controllable Language Models**

可控语言模型的指令阶梯推理 cs.CL

**SubmitDate**: 2025-11-13    [abs](http://arxiv.org/abs/2511.04694v2) [paper-pdf](None)

**Authors**: Zishuo Zheng, Vidhisha Balachandran, Chan Young Park, Faeze Brahman, Sachin Kumar

**Abstract**: As large language model (LLM) based systems take on high-stakes roles in real-world decision-making, they must reconcile competing instructions from multiple sources (e.g., model developers, users, and tools) within a single prompt context. Thus, enforcing an instruction hierarchy (IH) in LLMs, where higher-level directives override lower-priority requests, is critical for the reliability and controllability of LLMs. In this work, we reframe instruction hierarchy resolution as a reasoning task. Specifically, the model must first "think" about the relationship between a given user prompt and higher-priority (system) instructions before generating a response. To enable this capability via training, we construct VerIH, an instruction hierarchy dataset of constraint-following tasks with verifiable answers. This dataset comprises both aligned and conflicting system-user instructions. We show that lightweight reinforcement learning with VerIH effectively transfers general reasoning capabilities of models to instruction prioritization. Our finetuned models achieve consistent improvements on instruction following and instruction hierarchy benchmarks. This reasoning ability also generalizes to safety-critical settings beyond the training distribution. By treating safety issues as resolving conflicts between adversarial user inputs and predefined higher-priority policies, our trained model enhances robustness against jailbreak and prompt injection attacks. These results demonstrate that reasoning over instruction hierarchies provides a practical path to reliable LLMs, where updates to system prompts yield controllable and robust changes in model behavior.

摘要: 随着基于大型语言模型（LLM）的系统在现实世界的决策中扮演着高风险的角色，它们必须协调来自多个来源的竞争指令（例如，模型开发人员、用户和工具）在单个提示上下文中。因此，在LLM中强制执行指令层次结构（IHS）（其中更高级的指令优先于较低优先级的请求）对于LLM的可靠性和可控性至关重要。在这项工作中，我们将指令层次结构分解重新构建为一项推理任务。具体来说，模型必须在生成响应之前首先“思考”给定用户提示和更高优先级（系统）指令之间的关系。为了通过训练实现这种能力，我们构建了VerIHS，这是一个具有可验证答案的约束遵循任务的指令层次数据集。此数据集包括对齐和冲突的系统用户指令。我们表明，使用VerIHS的轻量级强化学习可以有效地将模型的一般推理能力转移到指令优先级。我们的微调模型在指令遵循和指令层次基准方面实现了一致的改进。这种推理能力还推广到培训分布以外的安全关键环境。通过将安全问题视为解决敌对用户输入和预定义的高优先级策略之间的冲突，我们训练的模型增强了针对越狱和即时注入攻击的鲁棒性。这些结果表明，对指令层次结构的推理提供了一条通往可靠LLM的实用途径，其中对系统提示的更新会产生模型行为的可控且稳健的变化。



## **23. SecInfer: Preventing Prompt Injection via Inference-time Scaling**

SecInfer：通过推理时缩放防止提示注入 cs.CR

**SubmitDate**: 2025-11-13    [abs](http://arxiv.org/abs/2509.24967v3) [paper-pdf](None)

**Authors**: Yupei Liu, Yanting Wang, Yuqi Jia, Jinyuan Jia, Neil Zhenqiang Gong

**Abstract**: Prompt injection attacks pose a pervasive threat to the security of Large Language Models (LLMs). State-of-the-art prevention-based defenses typically rely on fine-tuning an LLM to enhance its security, but they achieve limited effectiveness against strong attacks. In this work, we propose \emph{SecInfer}, a novel defense against prompt injection attacks built on \emph{inference-time scaling}, an emerging paradigm that boosts LLM capability by allocating more compute resources for reasoning during inference. SecInfer consists of two key steps: \emph{system-prompt-guided sampling}, which generates multiple responses for a given input by exploring diverse reasoning paths through a varied set of system prompts, and \emph{target-task-guided aggregation}, which selects the response most likely to accomplish the intended task. Extensive experiments show that, by leveraging additional compute at inference, SecInfer effectively mitigates both existing and adaptive prompt injection attacks, outperforming state-of-the-art defenses as well as existing inference-time scaling approaches.

摘要: 提示注入攻击对大型语言模型（LLM）的安全性构成普遍威胁。最先进的基于预防的防御通常依赖于对LLM进行微调来增强其安全性，但它们对强攻击的有效性有限。在这项工作中，我们提出了\{SecInfer}，这是一种基于\{推断时间缩放}的新型防御方法，这是一种新兴范式，通过在推断期间分配更多计算资源进行推理来增强LLM能力。SecInfer由两个关键步骤组成：\{系统提示引导采样}，通过通过不同的系统提示集探索不同的推理路径，为给定输入生成多个响应，以及\{target-task-guided aggregage}，选择最有可能完成预期任务的响应。大量实验表明，通过在推理时利用额外的计算，SecInfer有效地减轻了现有的和自适应的即时注入攻击，性能优于最先进的防御以及现有的推理时扩展方法。



## **24. Backdoor Attacks Against Speech Language Models**

针对语音语言模型的后门攻击 cs.CL

**SubmitDate**: **NEW** 2025-11-14    [abs](http://arxiv.org/abs/2510.01157v2) [paper-pdf](None)

**Authors**: Alexandrine Fortier, Thomas Thebaud, Jesús Villalba, Najim Dehak, Patrick Cardinal

**Abstract**: Large Language Models (LLMs) and their multimodal extensions are becoming increasingly popular. One common approach to enable multimodality is to cascade domain-specific encoders with an LLM, making the resulting model inherit vulnerabilities from all of its components. In this work, we present the first systematic study of audio backdoor attacks against speech language models. We demonstrate its effectiveness across four speech encoders and three datasets, covering four tasks: automatic speech recognition (ASR), speech emotion recognition, and gender and age prediction. The attack consistently achieves high success rates, ranging from 90.76% to 99.41%. To better understand how backdoors propagate, we conduct a component-wise analysis to identify the most vulnerable stages of the pipeline. Finally, we propose a fine-tuning-based defense that mitigates the threat of poisoned pretrained encoders.

摘要: 大型语言模型（LLM）及其多模式扩展正变得越来越受欢迎。启用多模式的一种常见方法是将特定于域的编码器与LLM级联，使生成的模型继承其所有组件的漏洞。在这项工作中，我们首次对针对语音语言模型的音频后门攻击进行了系统研究。我们在四个语音编码器和三个数据集中展示了它的有效性，涵盖四项任务：自动语音识别（ASB）、语音情感识别以及性别和年龄预测。该攻击的成功率始终很高，范围从90.76%到99.41%。为了更好地了解后门如何传播，我们进行了组件级分析，以识别管道中最脆弱的阶段。最后，我们提出了一种基于微调的防御，可以减轻中毒的预训练编码器的威胁。



## **25. EduGuardBench: A Holistic Benchmark for Evaluating the Pedagogical Fidelity and Adversarial Safety of LLMs as Simulated Teachers**

EduGuardBench：评估LLM作为模拟教师的教学忠实性和对抗安全性的整体基准 cs.CL

22 pages, 9 figures, accepted by AAAI2026 as oral paper

**SubmitDate**: 2025-11-11    [abs](http://arxiv.org/abs/2511.06890v1) [paper-pdf](None)

**Authors**: Yilin Jiang, Mingzi Zhang, Xuanyu Yin, Sheng Jin, Suyu Lu, Zuocan Ying, Zengyi Yu, Xiangjie Kong

**Abstract**: Large Language Models for Simulating Professions (SP-LLMs), particularly as teachers, are pivotal for personalized education. However, ensuring their professional competence and ethical safety is a critical challenge, as existing benchmarks fail to measure role-playing fidelity or address the unique teaching harms inherent in educational scenarios. To address this, we propose EduGuardBench, a dual-component benchmark. It assesses professional fidelity using a Role-playing Fidelity Score (RFS) while diagnosing harms specific to the teaching profession. It also probes safety vulnerabilities using persona-based adversarial prompts targeting both general harms and, particularly, academic misconduct, evaluated with metrics including Attack Success Rate (ASR) and a three-tier Refusal Quality assessment. Our extensive experiments on 14 leading models reveal a stark polarization in performance. While reasoning-oriented models generally show superior fidelity, incompetence remains the dominant failure mode across most models. The adversarial tests uncovered a counterintuitive scaling paradox, where mid-sized models can be the most vulnerable, challenging monotonic safety assumptions. Critically, we identified a powerful Educational Transformation Effect: the safest models excel at converting harmful requests into teachable moments by providing ideal Educational Refusals. This capacity is strongly negatively correlated with ASR, revealing a new dimension of advanced AI safety. EduGuardBench thus provides a reproducible framework that moves beyond siloed knowledge tests toward a holistic assessment of professional, ethical, and pedagogical alignment, uncovering complex dynamics essential for deploying trustworthy AI in education. See https://github.com/YL1N/EduGuardBench for Materials.

摘要: 大型语言模拟学习模型（SP-LLM），特别是作为教师，是个性化教育的关键。然而，确保他们的专业能力和道德安全是一个关键的挑战，因为现有的基准无法衡量角色扮演的忠诚度或解决教育场景中固有的独特教学危害。为了解决这个问题，我们提出了EduGuardBench，一个双组件基准。它使用角色扮演忠诚度评分（RFS）评估专业忠诚度，同时诊断针对教师职业的伤害。它还使用基于人物的对抗提示来调查安全漏洞，针对一般伤害，特别是学术不当行为，并使用攻击成功率（SVR）和三层拒绝质量评估等指标进行评估。我们对14种领先型号的广泛实验揭示了性能的明显两极分化。虽然以推理为导向的模型通常表现出卓越的保真度，但无能仍然是大多数模型的主要失败模式。对抗性测试揭示了一个违反直觉的缩放悖论，其中中型模型可能是最脆弱、最具挑战性的单调安全假设。至关重要的是，我们发现了一种强大的教育转型效应：最安全的模型擅长通过提供理想的教育拒绝将有害请求转化为可教的时刻。这种能力与ASB呈强烈负相关，揭示了高级人工智能安全性的新维度。因此，EduGuardBench提供了一个可重复的框架，该框架超越了孤立的知识测试，转向对专业、道德和教学一致性的全面评估，揭示了在教育中部署值得信赖的人工智能至关重要的复杂动态。请参阅https://github.com/YL1N/EduGuardBench了解材料。



## **26. SAFENLIDB: A Privacy-Preserving Safety Alignment Framework for LLM-based Natural Language Database Interfaces**

SAFENLIDB：基于LLM的自然语言数据库接口的保护隐私的安全对齐框架 cs.CL

AAAI 2026 Extended Version

**SubmitDate**: 2025-11-12    [abs](http://arxiv.org/abs/2511.06778v2) [paper-pdf](None)

**Authors**: Ruiheng Liu, XiaoBing Chen, Jinyu Zhang, Qiongwen Zhang, Yu Zhang, Bailong Yang

**Abstract**: The rapid advancement of Large Language Models (LLMs) has driven significant progress in Natural Language Interface to Database (NLIDB). However, the widespread adoption of LLMs has raised critical privacy and security concerns. During interactions, LLMs may unintentionally expose confidential database contents or be manipulated by attackers to exfiltrate data through seemingly benign queries. While current efforts typically rely on rule-based heuristics or LLM agents to mitigate this leakage risk, these methods still struggle with complex inference-based attacks, suffer from high false positive rates, and often compromise the reliability of SQL queries. To address these challenges, we propose \textsc{SafeNlidb}, a novel privacy-security alignment framework for LLM-based NLIDB. The framework features an automated pipeline that generates hybrid chain-of-thought interaction data from scratch, seamlessly combining implicit security reasoning with SQL generation. Additionally, we introduce reasoning warm-up and alternating preference optimization to overcome the multi-preference oscillations of Direct Preference Optimization (DPO), enabling LLMs to produce security-aware SQL through fine-grained reasoning without the need for human-annotated preference data. Extensive experiments demonstrate that our method outperforms both larger-scale LLMs and ideal-setting baselines, achieving significant security improvements while preserving high utility. WARNING: This work may contain content that is offensive and harmful!

摘要: 大型语言模型（LLM）的快速发展推动了自然语言数据库接口（NLIDB）的重大进展。然而，LLM的广泛采用引发了严重的隐私和安全问题。在交互过程中，LLM可能会无意中暴露机密的数据库内容，或者被攻击者操纵以通过看似良性的查询来泄露数据。虽然当前的工作通常依赖于基于规则的启发式方法或LLM代理来减轻这种泄露风险，但这些方法仍然难以应对复杂的基于推理的攻击，存在很高的假阳性率，并且经常损害SQL查询的可靠性。为了应对这些挑战，我们提出了\textsk {SafeNlidb}，这是一个针对基于LLM的NLIDB的新型隐私安全协调框架。该框架具有一个自动化管道，可以从头开始生成混合思想链交互数据，无缝地结合隐式安全推理与SQL生成。此外，我们引入了推理热身和交替偏好优化，以克服直接偏好优化（DPO）的多偏好振荡，使LLM能够通过细粒度推理生成安全感知SQL，而无需人工注释的偏好数据。大量实验表明，我们的方法优于大规模LLM和理想设置基线，在保持高实用性的同时实现了显着的安全改进。警告：本作品可能包含冒犯性和有害的内容！



## **27. CoSPED: Consistent Soft Prompt Targeted Data Extraction and Defense**

CoSPP：一致的软提示有针对性的数据提取和防御 cs.CR

**SubmitDate**: 2025-11-12    [abs](http://arxiv.org/abs/2510.11137v2) [paper-pdf](None)

**Authors**: Zhuochen Yang, Kar Wai Fok, Vrizlynn L. L. Thing

**Abstract**: Large language models have gained widespread attention recently, but their potential security vulnerabilities, especially privacy leakage, are also becoming apparent. To test and evaluate for data extraction risks in LLM, we proposed CoSPED, short for Consistent Soft Prompt targeted data Extraction and Defense. We introduce several innovative components, including Dynamic Loss, Additive Loss, Common Loss, and Self Consistency Decoding Strategy, and tested to enhance the consistency of the soft prompt tuning process. Through extensive experimentation with various combinations, we achieved an extraction rate of 65.2% at a 50-token prefix comparison. Our comparisons of CoSPED with other reference works confirm our superior extraction rates. We evaluate CoSPED on more scenarios, achieving Pythia model extraction rate of 51.7% and introducing cross-model comparison. Finally, we explore defense through Rank-One Model Editing and achieve a reduction in the extraction rate to 1.6%, which proves that our analysis of extraction mechanisms can directly inform effective mitigation strategies against soft prompt-based attacks.

摘要: 大型语言模型近年来得到了广泛的关注，但其潜在的安全漏洞，特别是隐私泄露，也越来越明显。为了测试和评估LLM中的数据提取风险，我们提出了CoSPED，即一致性软提示目标数据提取和防御的缩写。我们引入了几个创新的组件，包括动态损失，附加损失，共同损失，和自一致性解码策略，并测试，以提高软提示调整过程的一致性。通过对各种组合的广泛实验，我们在50个令牌前缀比较时实现了65.2%的提取率。我们将CoSPP与其他参考作品进行比较，证实了我们优越的提取率。我们在更多场景下评估了CoSPP，Pythia模型提取率达到51.7%，并引入了跨模型比较。最后，我们通过排名一模型编辑探索防御，并将提取率降低至1.6%，这证明我们对提取机制的分析可以直接为针对基于软预算的攻击的有效缓解策略提供信息。



## **28. CyberSOCEval: Benchmarking LLMs Capabilities for Malware Analysis and Threat Intelligence Reasoning**

CyberSOCEval：对LLM恶意软件分析和威胁情报推理的能力进行基准测试 cs.CR

**SubmitDate**: 2025-11-12    [abs](http://arxiv.org/abs/2509.20166v2) [paper-pdf](None)

**Authors**: Lauren Deason, Adam Bali, Ciprian Bejean, Diana Bolocan, James Crnkovich, Ioana Croitoru, Krishna Durai, Chase Midler, Calin Miron, David Molnar, Brad Moon, Bruno Ostarcevic, Alberto Peltea, Matt Rosenberg, Catalin Sandu, Arthur Saputkin, Sagar Shah, Daniel Stan, Ernest Szocs, Shengye Wan, Spencer Whitman, Sven Krasser, Joshua Saxe

**Abstract**: Today's cyber defenders are overwhelmed by a deluge of security alerts, threat intelligence signals, and shifting business context, creating an urgent need for AI systems to enhance operational security work. While Large Language Models (LLMs) have the potential to automate and scale Security Operations Center (SOC) operations, existing evaluations do not fully assess the scenarios most relevant to real-world defenders. This lack of informed evaluation impacts both AI developers and those applying LLMs to SOC automation. Without clear insight into LLM performance in real-world security scenarios, developers lack a north star for development, and users cannot reliably select the most effective models. Meanwhile, malicious actors are using AI to scale cyber attacks, highlighting the need for open source benchmarks to drive adoption and community-driven improvement among defenders and model developers. To address this, we introduce CyberSOCEval, a new suite of open source benchmarks within CyberSecEval 4. CyberSOCEval includes benchmarks tailored to evaluate LLMs in two tasks: Malware Analysis and Threat Intelligence Reasoning--core defensive domains with inadequate coverage in current benchmarks. Our evaluations show that larger, more modern LLMs tend to perform better, confirming the training scaling laws paradigm. We also find that reasoning models leveraging test time scaling do not achieve the same boost as in coding and math, suggesting these models have not been trained to reason about cybersecurity analysis, and pointing to a key opportunity for improvement. Finally, current LLMs are far from saturating our evaluations, showing that CyberSOCEval presents a significant challenge for AI developers to improve cyber defense capabilities.

摘要: 当今的网络防御者被大量安全警报、威胁情报信号和不断变化的业务环境所淹没，迫切需要人工智能系统来增强运营安全工作。虽然大型语言模型（LLM）有潜力自动化和扩展安全运营中心（SOC）操作，但现有的评估并未完全评估与现实世界的防御者最相关的场景。这种缺乏知情评估的情况影响了人工智能开发人员和将LLM应用于SOC自动化的人员。如果不清楚地了解现实安全场景中的LLM性能，开发人员缺乏开发北极星，用户也无法可靠地选择最有效的模型。与此同时，恶意行为者正在使用人工智能来扩大网络攻击规模，这凸显了开源基准的必要性，以推动防御者和模型开发者的采用和社区驱动的改进。为了解决这个问题，我们引入了CyberSOCEval，这是CyberSecEval 4中的一套新开源基准测试。CyberSOCEval包括为评估LLM两项任务而定制的基准：恶意软件分析和威胁情报推理--当前基准覆盖范围不足的核心防御领域。我们的评估表明，更大、更现代的LLM往往表现得更好，证实了训练缩放定律范式。我们还发现，利用测试时间扩展的推理模型并没有实现与编码和数学相同的提升，这表明这些模型尚未经过网络安全分析推理的训练，并指出了一个关键的改进机会。最后，当前的LLM远未饱和我们的评估，这表明CyberSOCEval对人工智能开发人员提高网络防御能力提出了重大挑战。



## **29. Cost-Minimized Label-Flipping Poisoning Attack to LLM Alignment**

对LLM对齐的成本最小化标签翻转中毒攻击 cs.LG

accepted for AAAI 2026 Special Track on AI Alignment

**SubmitDate**: 2025-11-13    [abs](http://arxiv.org/abs/2511.09105v1) [paper-pdf](None)

**Authors**: Shigeki Kusaka, Keita Saito, Mikoto Kudo, Takumi Tanabe, Akifumi Wachi, Youhei Akimoto

**Abstract**: Large language models (LLMs) are increasingly deployed in real-world systems, making it critical to understand their vulnerabilities. While data poisoning attacks during RLHF/DPO alignment have been studied empirically, their theoretical foundations remain unclear. We investigate the minimum-cost poisoning attack required to steer an LLM's policy toward an attacker's target by flipping preference labels during RLHF/DPO, without altering the compared outputs. We formulate this as a convex optimization problem with linear constraints, deriving lower and upper bounds on the minimum attack cost. As a byproduct of this theoretical analysis, we show that any existing label-flipping attack can be post-processed via our proposed method to reduce the number of label flips required while preserving the intended poisoning effect. Empirical results demonstrate that this cost-minimization post-processing can significantly reduce poisoning costs over baselines, particularly when the reward model's feature dimension is small relative to the dataset size. These findings highlight fundamental vulnerabilities in RLHF/DPO pipelines and provide tools to evaluate their robustness against low-cost poisoning attacks.

摘要: 大型语言模型（LLM）越来越多地部署在现实世界的系统中，因此了解其漏洞至关重要。虽然已经对WLHF/DPO对齐期间的数据中毒攻击进行了经验研究，但其理论基础仍不清楚。我们调查了通过在WLHF/DPO期间翻转偏好标签将LLM的政策引导到攻击者的目标所需的最小成本中毒攻击，而不改变比较的输出。我们将其表述为具有线性约束的凸优化问题，并推导出最小攻击成本的下限和上限。作为这一理论分析的副产品，我们表明，任何现有的标签翻转攻击都可以通过我们提出的方法进行后处理，以减少所需的标签翻转数量，同时保留预期的中毒效应。经验结果表明，这种成本最小化后处理可以显着降低基线上的中毒成本，特别是当奖励模型的特征维度相对于数据集大小较小时。这些发现凸显了WLHF/DPO管道中的基本漏洞，并提供了评估其针对低成本中毒攻击的稳健性的工具。



## **30. Joint-GCG: Unified Gradient-Based Poisoning Attacks on Retrieval-Augmented Generation Systems**

Joint-GCG：对检索增强生成系统的统一基于对象的中毒攻击 cs.CR

**SubmitDate**: 2025-11-13    [abs](http://arxiv.org/abs/2506.06151v2) [paper-pdf](None)

**Authors**: Haowei Wang, Rupeng Zhang, Junjie Wang, Mingyang Li, Yuekai Huang, Dandan Wang, Qing Wang

**Abstract**: Retrieval-Augmented Generation (RAG) systems enhance Large Language Models (LLMs) by retrieving relevant documents from external corpora before generating responses. This approach significantly expands LLM capabilities by leveraging vast, up-to-date external knowledge. However, this reliance on external knowledge makes RAG systems vulnerable to corpus poisoning attacks that manipulate generated outputs via poisoned document injection. Existing poisoning attack strategies typically treat the retrieval and generation stages as disjointed, limiting their effectiveness. We propose Joint-GCG, the first framework to unify gradient-based attacks across both retriever and generator models through three innovations: (1) Cross-Vocabulary Projection for aligning embedding spaces, (2) Gradient Tokenization Alignment for synchronizing token-level gradient signals, and (3) Adaptive Weighted Fusion for dynamically balancing attacking objectives. Evaluations demonstrate that Joint-GCG achieves at most 25% and an average of 5% higher attack success rate than previous methods across multiple retrievers and generators. While optimized under a white-box assumption, the generated poisons show unprecedented transferability to unseen models. Joint-GCG's innovative unification of gradient-based attacks across retrieval and generation stages fundamentally reshapes our understanding of vulnerabilities within RAG systems. Our code is available at https://github.com/NicerWang/Joint-GCG.

摘要: 检索增强生成（RAG）系统通过在生成响应之前从外部库检索相关文档来增强大型语言模型（LLM）。这种方法通过利用大量、最新的外部知识来显着扩展LLM能力。然而，这种对外部知识的依赖使得RAG系统容易受到通过有毒文档注入来操纵生成的输出的数据库中毒攻击。现有的中毒攻击策略通常将检索和生成阶段视为脱节的，从而限制了它们的有效性。我们提出了Joint-GCG，这是第一个通过三项创新统一检索器和生成器模型中基于梯度的攻击的框架：（1）用于对齐嵌入空间的跨词汇投影，（2）用于同步标记级梯度信号的梯度令牌化对齐，以及（3）用于动态平衡攻击目标的自适应加权融合。评估表明，Joint-GCG在多个检索器和生成器上的攻击成功率比以前的方法最多高25%，平均高5%。虽然在白盒假设下进行了优化，但生成的毒药显示出前所未有的可转移性，以转移到未见过的模型。Joint-GCG创新地统一了检索和生成阶段的基于梯度的攻击，从根本上重塑了我们对RAG系统中漏洞的理解。我们的代码可以在https://github.com/NicerWang/Joint-GCG上找到。



## **31. Siren: A Learning-Based Multi-Turn Attack Framework for Simulating Real-World Human Jailbreak Behaviors**

Siren：一个基于学习的多回合攻击框架，用于模拟现实世界的人类越狱行为 cs.CL

Accepted at ACSAC 2025

**SubmitDate**: **NEW** 2025-11-14    [abs](http://arxiv.org/abs/2501.14250v2) [paper-pdf](None)

**Authors**: Yi Zhao, Youzhi Zhang

**Abstract**: Large language models (LLMs) are widely used in real-world applications, raising concerns about their safety and trustworthiness. While red-teaming with jailbreak prompts exposes the vulnerabilities of LLMs, current efforts focus primarily on single-turn attacks, overlooking the multi-turn strategies used by real-world adversaries. Existing multi-turn methods rely on static patterns or predefined logical chains, failing to account for the dynamic strategies during attacks. We propose Siren, a learning-based multi-turn attack framework designed to simulate real-world human jailbreak behaviors. Siren consists of three stages: (1) MiniMax-driven training set construction utilizing Turn-Level LLM feedback, (2) post-training attackers with supervised fine-tuning (SFT) and direct preference optimization (DPO), and (3) interactions between the attacking and target LLMs. Experiments demonstrate that Siren achieves an attack success rate (ASR) of 90% with LLaMA-3-8B as the attacker against Gemini-1.5-Pro as the target model, and 70% with Mistral-7B against GPT-4o, significantly outperforming single-turn baselines. Moreover, Siren with a 7B-scale model achieves performance comparable to a multi-turn baseline that leverages GPT-4o as the attacker, while requiring fewer turns and employing decomposition strategies that are better semantically aligned with attack goals. We hope Siren inspires the development of stronger defenses against advanced multi-turn jailbreak attacks under realistic scenarios. Code is available at https://github.com/YiyiyiZhao/siren. Warning: This paper contains potentially harmful text.

摘要: 大型语言模型（LLM）广泛应用于现实世界的应用程序中，引发了对其安全性和可信性的担忧。虽然与越狱提示进行红色合作暴露了LLM的脆弱性，但目前的工作主要集中在单回合攻击上，忽视了现实世界对手使用的多回合策略。现有的多回合方法依赖于静态模式或预定义的逻辑链，未能考虑攻击期间的动态策略。我们提出了Siren，这是一个基于学习的多回合攻击框架，旨在模拟现实世界中的人类越狱行为。Siren由三个阶段组成：（1）利用Turn-Level LLM反馈的MiniMax驱动的训练集构建，（2）训练后的攻击者进行监督微调（SFT）和直接偏好优化（DPO），以及（3）攻击和目标LLM之间的交互。实验表明，Siren以LLaMA-3-8B为攻击者，对Gemini-1.5-Pro为目标模型的攻击成功率（ASR）为90%，Mistral-7 B对GPT-4 o的攻击成功率为70%，明显优于单回合基线。此外，具有7 B规模模型的Siren实现了与利用GPT-4 o作为攻击者的多回合基线相当的性能，同时需要更少的回合并采用更好地与攻击目标语义一致的分解策略。我们希望Siren能够激发开发更强大的防御，以对抗现实场景下的高级多回合越狱攻击。代码可在https://github.com/YiyiyiZhao/siren上获取。警告：本文包含潜在有害的文本。



## **32. Secure Retrieval-Augmented Generation against Poisoning Attacks**

针对中毒攻击的安全检索增强生成 cs.CR

To appear in IEEE BigData 2025

**SubmitDate**: 2025-11-11    [abs](http://arxiv.org/abs/2510.25025v2) [paper-pdf](None)

**Authors**: Zirui Cheng, Jikai Sun, Anjun Gao, Yueyang Quan, Zhuqing Liu, Xiaohua Hu, Minghong Fang

**Abstract**: Large language models (LLMs) have transformed natural language processing (NLP), enabling applications from content generation to decision support. Retrieval-Augmented Generation (RAG) improves LLMs by incorporating external knowledge but also introduces security risks, particularly from data poisoning, where the attacker injects poisoned texts into the knowledge database to manipulate system outputs. While various defenses have been proposed, they often struggle against advanced attacks. To address this, we introduce RAGuard, a detection framework designed to identify poisoned texts. RAGuard first expands the retrieval scope to increase the proportion of clean texts, reducing the likelihood of retrieving poisoned content. It then applies chunk-wise perplexity filtering to detect abnormal variations and text similarity filtering to flag highly similar texts. This non-parametric approach enhances RAG security, and experiments on large-scale datasets demonstrate its effectiveness in detecting and mitigating poisoning attacks, including strong adaptive attacks.

摘要: 大型语言模型（LLM）改变了自然语言处理（NLP），使从内容生成到决策支持的应用程序成为可能。检索增强生成（RAG）通过整合外部知识来改进LLM，但也会引入安全风险，特别是来自数据中毒的风险，即攻击者将有毒文本注入知识数据库以操纵系统输出。虽然已经提出了各种防御措施，但它们常常难以抵御高级攻击。为了解决这个问题，我们引入了RAGuard，这是一个旨在识别有毒文本的检测框架。RAGuard首先扩大检索范围，增加干净文本的比例，降低检索有毒内容的可能性。然后，它应用块式困惑过滤来检测异常变化，并应用文本相似性过滤来标记高度相似的文本。这种非参数方法增强了RAG安全性，大规模数据集上的实验证明了其在检测和减轻中毒攻击（包括强适应性攻击）方面的有效性。



## **33. HumorReject: Decoupling LLM Safety from Refusal Prefix via A Little Humor**

幽默：通过一点幽默将LLM安全与拒绝前置脱钩 cs.LG

**SubmitDate**: 2025-11-11    [abs](http://arxiv.org/abs/2501.13677v3) [paper-pdf](None)

**Authors**: Zihui Wu, Haichang Gao, Jiacheng Luo, Zhaoxiang Liu

**Abstract**: Large Language Models (LLMs) commonly rely on explicit refusal prefixes for safety, making them vulnerable to prefix injection attacks. We introduce HumorReject, a novel data-driven approach that reimagines LLM safety by decoupling it from refusal prefixes through humor as an indirect refusal strategy. Rather than explicitly rejecting harmful instructions, HumorReject responds with contextually appropriate humor that naturally defuses potentially dangerous requests. Our approach effectively addresses common "over-defense" issues while demonstrating superior robustness against various attack vectors. Our findings suggest that improvements in training data design can be as important as the alignment algorithm itself in achieving effective LLM safety. The code and dataset are available at https://github.com/wooozihui/HumorReject.

摘要: 大型语言模型（LLM）通常依赖于显式拒绝前缀来保证安全，这使得它们容易受到前缀注入攻击。我们引入了幽默感，这是一种新颖的数据驱动方法，通过幽默将其与拒绝开头脱钩，重新构想了LLM的安全性，将其作为一种间接拒绝策略。幽默感并没有明确拒绝有害的指令，而是以符合上下文的幽默来回应，从而自然地化解潜在危险的请求。我们的方法有效地解决了常见的“过度防御”问题，同时展示了针对各种攻击载体的卓越鲁棒性。我们的研究结果表明，在实现有效的LLM安全性方面，训练数据设计的改进与对齐算法本身一样重要。代码和数据集可在https://github.com/wooozihui/HumorReject上获得。



## **34. MENTOR: A Metacognition-Driven Self-Evolution Framework for Uncovering and Mitigating Implicit Risks in LLMs on Domain Tasks**

MENTOR：一个元认知驱动的自我进化框架，用于发现和缓解领域任务LLM中的隐性风险 cs.AI

**SubmitDate**: 2025-11-11    [abs](http://arxiv.org/abs/2511.07107v1) [paper-pdf](None)

**Authors**: Liang Shan, Kaicheng Shen, Wen Wu, Zhenyu Ying, Chaochao Lu, Guangze Ye, Liang He

**Abstract**: Ensuring the safety and value alignment of large language models (LLMs) is critical for their deployment. Current alignment efforts primarily target explicit risks such as bias, hate speech, and violence. However, they often fail to address deeper, domain-specific implicit risks and lack a flexible, generalizable framework applicable across diverse specialized fields. Hence, we proposed MENTOR: A MEtacognition-driveN self-evoluTion framework for uncOvering and mitigating implicit Risks in LLMs on Domain Tasks. To address the limitations of labor-intensive human evaluation, we introduce a novel metacognitive self-assessment tool. This enables LLMs to reflect on potential value misalignments in their responses using strategies like perspective-taking and consequential thinking. We also release a supporting dataset of 9,000 risk queries spanning education, finance, and management to enhance domain-specific risk identification. Subsequently, based on the outcomes of metacognitive reflection, the framework dynamically generates supplementary rule knowledge graphs that extend predefined static rule trees. This enables models to actively apply validated rules to future similar challenges, establishing a continuous self-evolution cycle that enhances generalization by reducing maintenance costs and inflexibility of static systems. Finally, we employ activation steering during inference to guide LLMs in following the rules, a cost-effective method to robustly enhance enforcement across diverse contexts. Experimental results show MENTOR's effectiveness: In defensive testing across three vertical domains, the framework substantially reduces semantic attack success rates, enabling a new level of implicit risk mitigation for LLMs. Furthermore, metacognitive assessment not only aligns closely with baseline human evaluators but also delivers more thorough and insightful analysis of LLMs value alignment.

摘要: 确保大型语言模型（LLM）的安全性和价值一致性对于它们的部署至关重要。当前的协调工作主要针对偏见、仇恨言论和暴力等明显风险。然而，它们往往无法解决更深层次的、特定领域的隐性风险，并且缺乏适用于不同专业领域的灵活、可概括的框架。因此，我们提出了MENTOR：一个元认知驱动自我进化框架，用于应对和减轻域任务上LLM中的隐性风险。为了解决劳动密集型人类评估的局限性，我们引入了一种新型的元认知自我评估工具。这使得LLM能够使用观点审视和后果性思维等策略来反思其应对措施中潜在的价值失调。我们还发布了包含9，000个涵盖教育、财务和管理的风险查询的支持数据集，以增强特定领域的风险识别。随后，基于元认知反射的结果，框架动态生成扩展预定义静态规则树的补充规则知识图。这使模型能够积极应用经过验证的规则来应对未来的类似挑战，建立持续的自我进化循环，通过降低维护成本和静态系统的灵活性来增强概括性。最后，我们在推理过程中使用激活引导来指导LLM遵守规则，这是一种具有成本效益的方法，可以在不同背景下强有力地增强执法。实验结果表明了MENTOR的有效性：在跨三个垂直领域的防御测试中，该框架大幅降低了语义攻击成功率，使LLM的隐性风险缓解达到了新水平。此外，元认知评估不仅与基线人类评估者密切一致，而且还对LLM价值一致性提供更彻底、更有洞察力的分析。



## **35. KG-DF: A Black-box Defense Framework against Jailbreak Attacks Based on Knowledge Graphs**

KG-DF：基于知识图的越狱攻击黑匣子防御框架 cs.CR

**SubmitDate**: 2025-11-12    [abs](http://arxiv.org/abs/2511.07480v1) [paper-pdf](None)

**Authors**: Shuyuan Liu, Jiawei Chen, Xiao Yang, Hang Su, Zhaoxia Yin

**Abstract**: With the widespread application of large language models (LLMs) in various fields, the security challenges they face have become increasingly prominent, especially the issue of jailbreak. These attacks induce the model to generate erroneous or uncontrolled outputs through crafted inputs, threatening the generality and security of the model. Although existing defense methods have shown some effectiveness, they often struggle to strike a balance between model generality and security. Excessive defense may limit the normal use of the model, while insufficient defense may lead to security vulnerabilities. In response to this problem, we propose a Knowledge Graph Defense Framework (KG-DF). Specifically, because of its structured knowledge representation and semantic association capabilities, Knowledge Graph(KG) can be searched by associating input content with safe knowledge in the knowledge base, thus identifying potentially harmful intentions and providing safe reasoning paths. However, traditional KG methods encounter significant challenges in keyword extraction, particularly when confronted with diverse and evolving attack strategies. To address this issue, we introduce an extensible semantic parsing module, whose core task is to transform the input query into a set of structured and secure concept representations, thereby enhancing the relevance of the matching process. Experimental results show that our framework enhances defense performance against various jailbreak attack methods, while also improving the response quality of the LLM in general QA scenarios by incorporating domain-general knowledge.

摘要: 随着大型语言模型（LLM）在各个领域的广泛应用，其面临的安全挑战日益突出，尤其是越狱问题。这些攻击导致模型通过精心设计的输入生成错误或不受控的输出，威胁模型的通用性和安全性。尽管现有的防御方法已经表现出一定的有效性，但它们常常难以在模型通用性和安全性之间取得平衡。过度的防御可能会限制模型的正常使用，而防御不足可能会导致安全漏洞。为了应对这个问题，我们提出了一个知识图防御框架（KG-DF）。具体来说，由于知识图（KG）的结构化知识表示和语义关联能力，可以通过将输入内容与知识库中的安全知识相关联来搜索知识图（KG），从而识别潜在的有害意图并提供安全的推理路径。然而，传统的KG方法在关键词提取方面遇到了重大挑战，特别是当面临多样化且不断发展的攻击策略时。为了解决这个问题，我们引入了一个可扩展的语义解析模块，其核心任务是将输入查询转换为一组结构化且安全的概念表示，从而增强匹配过程的相关性。实验结果表明，我们的框架增强了针对各种越狱攻击方法的防御性能，同时还通过融入领域常识提高了LLM在一般QA场景中的响应质量。



## **36. UDora: A Unified Red Teaming Framework against LLM Agents by Dynamically Hijacking Their Own Reasoning**

UPora：通过动态劫持LLM代理自己的推理来对抗他们的统一红色团队框架 cs.CR

**SubmitDate**: 2025-11-13    [abs](http://arxiv.org/abs/2503.01908v3) [paper-pdf](None)

**Authors**: Jiawei Zhang, Shuang Yang, Bo Li

**Abstract**: Large Language Model (LLM) agents equipped with external tools have become increasingly powerful for complex tasks such as web shopping, automated email replies, and financial trading. However, these advancements amplify the risks of adversarial attacks, especially when agents can access sensitive external functionalities. Nevertheless, manipulating LLM agents into performing targeted malicious actions or invoking specific tools remains challenging, as these agents extensively reason or plan before executing final actions. In this work, we present UDora, a unified red teaming framework designed for LLM agents that dynamically hijacks the agent's reasoning processes to compel malicious behavior. Specifically, UDora first generates the model's reasoning trace for the given task, then automatically identifies optimal points within this trace to insert targeted perturbations. The resulting perturbed reasoning is then used as a surrogate response for optimization. By iteratively applying this process, the LLM agent will then be induced to undertake designated malicious actions or to invoke specific malicious tools. Our approach demonstrates superior effectiveness compared to existing methods across three LLM agent datasets. The code is available at https://github.com/AI-secure/UDora.

摘要: 配备外部工具的大型语言模型（LLM）代理在网络购物、自动电子邮件回复和金融交易等复杂任务中变得越来越强大。然而，这些进步放大了对抗攻击的风险，特别是当代理可以访问敏感的外部功能时。然而，操纵LLM代理执行有针对性的恶意操作或调用特定工具仍然具有挑战性，因为这些代理在执行最终操作之前进行了广泛的推理或计划。在这项工作中，我们介绍了UPora，这是一个为LLM代理设计的统一红色团队框架，它动态劫持代理的推理过程以迫使恶意行为。具体来说，UPora首先为给定任务生成模型的推理轨迹，然后自动识别此轨迹内的最佳点以插入有针对性的扰动。然后将产生的扰动推理用作优化的替代响应。通过迭代应用此过程，LLM代理将被诱导采取指定的恶意操作或调用特定的恶意工具。与三个LLM代理数据集的现有方法相比，我们的方法表现出了卓越的有效性。该代码可在https://github.com/AI-secure/UDora上获取。



## **37. MPMA: Preference Manipulation Attack Against Model Context Protocol**

MPMA：针对模型上下文协议的偏好操纵攻击 cs.CR

This is an extended version of the copyrighted publication at AAAI

**SubmitDate**: 2025-11-12    [abs](http://arxiv.org/abs/2505.11154v2) [paper-pdf](None)

**Authors**: Zihan Wang, Rui Zhang, Yu Liu, Wenshu Fan, Wenbo Jiang, Qingchuan Zhao, Hongwei Li, Guowen Xu

**Abstract**: Model Context Protocol (MCP) standardizes interface mapping for large language models (LLMs) to access external data and tools, which revolutionizes the paradigm of tool selection and facilitates the rapid expansion of the LLM agent tool ecosystem. However, as the MCP is increasingly adopted, third-party customized versions of the MCP server expose potential security vulnerabilities. In this paper, we first introduce a novel security threat, which we term the MCP Preference Manipulation Attack (MPMA). An attacker deploys a customized MCP server to manipulate LLMs, causing them to prioritize it over other competing MCP servers. This can result in economic benefits for attackers, such as revenue from paid MCP services or advertising income generated from free servers. To achieve MPMA, we first design a Direct Preference Manipulation Attack (DPMA) that achieves significant effectiveness by inserting the manipulative word and phrases into the tool name and description. However, such a direct modification is obvious to users and lacks stealthiness. To address these limitations, we further propose Genetic-based Advertising Preference Manipulation Attack (GAPMA). GAPMA employs four commonly used strategies to initialize descriptions and integrates a Genetic Algorithm (GA) to enhance stealthiness. The experiment results demonstrate that GAPMA balances high effectiveness and stealthiness. Our study reveals a critical vulnerability of the MCP in open ecosystems, highlighting an urgent need for robust defense mechanisms to ensure the fairness of the MCP ecosystem.

摘要: 模型上下文协议（HCP）将大型语言模型（LLM）的接口映射同步化，以访问外部数据和工具，这彻底改变了工具选择的范式，并促进了LLM代理工具生态系统的快速扩展。然而，随着LCP越来越多地采用，第三方定制版本的LCP服务器暴露了潜在的安全漏洞。在本文中，我们首先介绍了一种新型的安全威胁，我们将其称为LCP偏好操纵攻击（MPMA）。攻击者部署自定义的LCP服务器来操纵LLM，导致他们将其优先于其他竞争的LCP服务器。这可能会为攻击者带来经济利益，例如付费HCP服务的收入或免费服务器产生的广告收入。为了实现MPMA，我们首先设计了一种直接偏好操纵攻击（DPMA），该攻击通过将操纵性单词和短语插入工具名称和描述中来达到显着的效果。但这样的直接修改对于用户来说是显而易见的，缺乏隐蔽性。为了解决这些限制，我们进一步提出了基于基因的广告偏好操纵攻击（GAPMA）。GAPMA采用四种常用的策略来初始化描述，并集成了遗传算法（GA），以提高隐身性。实验结果表明，GAPMA算法在高效性和隐蔽性之间取得了较好的平衡.我们的研究揭示了开放生态系统中MCP的关键脆弱性，强调了迫切需要强大的防御机制，以确保MCP生态系统的公平性。



## **38. Hail to the Thief: Exploring Attacks and Defenses in Decentralised GRPO**

向小偷致敬：探索分散式GRPO中的攻击和防御 cs.LG

**SubmitDate**: **NEW** 2025-11-14    [abs](http://arxiv.org/abs/2511.09780v1) [paper-pdf](None)

**Authors**: Nikolay Blagoev, Oğuzhan Ersoy, Lydia Yiyu Chen

**Abstract**: Group Relative Policy Optimization (GRPO) has demonstrated great utilization in post-training of Large Language Models (LLMs). In GRPO, prompts are answered by the model and, through reinforcement learning, preferred completions are learnt. Owing to the small communication volume, GRPO is inherently suitable for decentralised training as the prompts can be concurrently answered by multiple nodes and then exchanged in the forms of strings. In this work, we present the first adversarial attack in decentralised GRPO. We demonstrate that malicious parties can poison such systems by injecting arbitrary malicious tokens in benign models in both out-of-context and in-context attacks. Using empirical examples of math and coding tasks, we show that adversarial attacks can easily poison the benign nodes, polluting their local LLM post-training, achieving attack success rates up to 100% in as few as 50 iterations. We propose two ways to defend against these attacks, depending on whether all users train the same model or different models. We show that these defenses can achieve stop rates of up to 100%, making the attack impossible.

摘要: 组相对策略优化（GRPO）在大型语言模型（LLM）的后训练中表现出了巨大的利用率。在GRPO中，模型回答提示，并通过强化学习首选的完成。由于通信量小，GRPO本质上适合去中心化训练，因为提示可以由多个节点同时回答，然后以字符串的形式交换。在这项工作中，我们展示了去中心化GRPO中的第一次对抗攻击。我们证明，恶意方可以在上下文外和上下文内攻击中通过在良性模型中注入任意恶意令牌来毒害此类系统。使用数学和编码任务的经验示例，我们表明对抗性攻击很容易毒害良性节点，污染其本地LLM后训练，在短短50次迭代中实现高达100%的攻击成功率。我们提出了两种防御这些攻击的方法，具体取决于所有用户是训练相同的模型还是不同的模型。我们表明，这些防御措施可以实现高达100%的停止率，使攻击变得不可能。



## **39. Graph of Attacks with Pruning: Optimizing Stealthy Jailbreak Prompt Generation for Enhanced LLM Content Moderation**

修剪攻击图：优化隐形越狱提示生成以增强LLM内容审核 cs.CR

14 pages, 5 figures; published in EMNLP 2025 ; Code at: https://github.com/dsbuddy/GAP-LLM-Safety

**SubmitDate**: **NEW** 2025-11-14    [abs](http://arxiv.org/abs/2501.18638v3) [paper-pdf](None)

**Authors**: Daniel Schwartz, Dmitriy Bespalov, Zhe Wang, Ninad Kulkarni, Yanjun Qi

**Abstract**: As large language models (LLMs) become increasingly prevalent, ensuring their robustness against adversarial misuse is crucial. This paper introduces the GAP (Graph of Attacks with Pruning) framework, an advanced approach for generating stealthy jailbreak prompts to evaluate and enhance LLM safeguards. GAP addresses limitations in existing tree-based LLM jailbreak methods by implementing an interconnected graph structure that enables knowledge sharing across attack paths. Our experimental evaluation demonstrates GAP's superiority over existing techniques, achieving a 20.8% increase in attack success rates while reducing query costs by 62.7%. GAP consistently outperforms state-of-the-art methods for attacking both open and closed LLMs, with attack success rates of >96%. Additionally, we present specialized variants like GAP-Auto for automated seed generation and GAP-VLM for multimodal attacks. GAP-generated prompts prove highly effective in improving content moderation systems, increasing true positive detection rates by 108.5% and accuracy by 183.6% when used for fine-tuning. Our implementation is available at https://github.com/dsbuddy/GAP-LLM-Safety.

摘要: 随着大型语言模型（LLM）变得越来越普遍，确保其针对对抗性滥用的鲁棒性至关重要。本文介绍了GAP（带有修剪的攻击图）框架，这是一种生成隐形越狱提示以评估和增强LLM保障措施的高级方法。GAP通过实现互连的图结构来解决现有基于树的LLM越狱方法的局限性，该结构能够实现跨攻击路径的知识共享。我们的实验评估证明了GAP相对于现有技术的优越性，攻击成功率提高了20.8%，同时将查询成本降低了62.7%。对于攻击开放式和封闭式LLM，RAP始终优于最先进的方法，攻击成功率> 96%。此外，我们还提供了专门的变体，例如用于自动种子生成的GAP-Auto和用于多模式攻击的GAP-VLM。事实证明，由间隙生成的提示在改进内容审核系统方面非常有效，用于微调时，真阳性检测率可提高108.5%，准确率可提高183.6%。我们的实施可在https://github.com/dsbuddy/GAP-LLM-Safety上获取。



## **40. Efficient LLM Safety Evaluation through Multi-Agent Debate**

通过多主体辩论进行高效的LLM安全评估 cs.AI

9 pages of main text, 14 pages total, 4 figures

**SubmitDate**: 2025-11-11    [abs](http://arxiv.org/abs/2511.06396v1) [paper-pdf](None)

**Authors**: Dachuan Lin, Guobin Shen, Zihao Yang, Tianrong Liu, Dongcheng Zhao, Yi Zeng

**Abstract**: Safety evaluation of large language models (LLMs) increasingly relies on LLM-as-a-Judge frameworks, but the high cost of frontier models limits scalability. We propose a cost-efficient multi-agent judging framework that employs Small Language Models (SLMs) through structured debates among critic, defender, and judge agents. To rigorously assess safety judgments, we construct HAJailBench, a large-scale human-annotated jailbreak benchmark comprising 12,000 adversarial interactions across diverse attack methods and target models. The dataset provides fine-grained, expert-labeled ground truth for evaluating both safety robustness and judge reliability. Our SLM-based framework achieves agreement comparable to GPT-4o judges on HAJailBench while substantially reducing inference cost. Ablation results show that three rounds of debate yield the optimal balance between accuracy and efficiency. These findings demonstrate that structured, value-aligned debate enables SLMs to capture semantic nuances of jailbreak attacks and that HAJailBench offers a reliable foundation for scalable LLM safety evaluation.

摘要: 大型语言模型（LLM）的安全评估越来越依赖于LLM作为法官框架，但前沿模型的高成本限制了可扩展性。我们提出了一个具有成本效益的多智能体判断框架，采用小语言模型（SLM）通过结构化的辩论之间的批评家，辩护人和法官代理。为了严格评估安全判断，我们构建了HAJailBench，这是一个大规模的人类注释的越狱基准测试，包括12，000个针对不同攻击方法和目标模型的对抗性交互。该数据集提供了细粒度的、专家标记的基础事实，用于评估安全鲁棒性和判断可靠性。我们基于LM的框架实现了与HAJailBench上GPT-4 o法官相当的一致，同时大幅降低了推理成本。消融结果表明，三轮辩论在准确性和效率之间取得了最佳平衡。这些发现表明，结构化、价值一致的辩论使STM能够捕捉越狱攻击的语义细微差别，并且HAJailBench为可扩展的LLM安全评估提供了可靠的基础。



## **41. Why does weak-OOD help? A Further Step Towards Understanding Jailbreaking VLMs**

为什么弱OOD有帮助？进一步了解越狱的VLM cs.CR

**SubmitDate**: 2025-11-12    [abs](http://arxiv.org/abs/2511.08367v1) [paper-pdf](None)

**Authors**: Yuxuan Zhou, Yuzhao Peng, Yang Bai, Kuofeng Gao, Yihao Zhang, Yechao Zhang, Xun Chen, Tao Yu, Tao Dai, Shu-Tao Xia

**Abstract**: Large Vision-Language Models (VLMs) are susceptible to jailbreak attacks: researchers have developed a variety of attack strategies that can successfully bypass the safety mechanisms of VLMs. Among these approaches, jailbreak methods based on the Out-of-Distribution (OOD) strategy have garnered widespread attention due to their simplicity and effectiveness. This paper further advances the in-depth understanding of OOD-based VLM jailbreak methods. Experimental results demonstrate that jailbreak samples generated via mild OOD strategies exhibit superior performance in circumventing the safety constraints of VLMs--a phenomenon we define as ''weak-OOD''. To unravel the underlying causes of this phenomenon, this study takes SI-Attack, a typical OOD-based jailbreak method, as the research object. We attribute this phenomenon to a trade-off between two dominant factors: input intent perception and model refusal triggering. The inconsistency in how these two factors respond to OOD manipulations gives rise to this phenomenon. Furthermore, we provide a theoretical argument for the inevitability of such inconsistency from the perspective of discrepancies between model pre-training and alignment processes. Building on the above insights, we draw inspiration from optical character recognition (OCR) capability enhancement--a core task in the pre-training phase of mainstream VLMs. Leveraging this capability, we design a simple yet highly effective VLM jailbreak method, whose performance outperforms that of SOTA baselines.

摘要: 大型视觉语言模型（VLM）容易受到越狱攻击：研究人员开发了多种攻击策略，可以成功绕过VLM的安全机制。在这些方法中，基于分发外（OOD）策略的越狱方法因其简单性和有效性而受到广泛关注。本文进一步深入了解基于OOD的VLM越狱方法。实验结果表明，通过温和OOD策略生成的越狱样本在规避VLM的安全约束方面表现出优异的性能--我们将这种现象定义为“弱OOD”。为了解开这种现象的根本原因，本研究以SI-Attack（一种典型的基于OOD的越狱方法）为研究对象。我们将这种现象归因于两个主要因素之间的权衡：输入意图感知和模型拒绝触发。这两个因素对OOD操纵的反应不一致导致了这种现象。此外，我们从模型预训练和对齐过程之间差异的角度为这种不一致性的不可避免性提供了理论论据。基于上述见解，我们从光学字符识别（OCR）能力增强中获得灵感-这是主流VLM预训练阶段的核心任务。利用这个能力，我们设计了一个简单而高效的VLM越狱方法，其性能优于SOTA基线。



## **42. Differentiated Directional Intervention A Framework for Evading LLM Safety Alignment**

差异化定向干预避免LLM安全一致的框架 cs.CR

AAAI-26-AIA

**SubmitDate**: 2025-11-12    [abs](http://arxiv.org/abs/2511.06852v2) [paper-pdf](None)

**Authors**: Peng Zhang, Peijie Sun

**Abstract**: Safety alignment instills in Large Language Models (LLMs) a critical capacity to refuse malicious requests. Prior works have modeled this refusal mechanism as a single linear direction in the activation space. We posit that this is an oversimplification that conflates two functionally distinct neural processes: the detection of harm and the execution of a refusal. In this work, we deconstruct this single representation into a Harm Detection Direction and a Refusal Execution Direction. Leveraging this fine-grained model, we introduce Differentiated Bi-Directional Intervention (DBDI), a new white-box framework that precisely neutralizes the safety alignment at critical layer. DBDI applies adaptive projection nullification to the refusal execution direction while suppressing the harm detection direction via direct steering. Extensive experiments demonstrate that DBDI outperforms prominent jailbreaking methods, achieving up to a 97.88\% attack success rate on models such as Llama-2. By providing a more granular and mechanistic framework, our work offers a new direction for the in-depth understanding of LLM safety alignment.

摘要: 安全一致为大型语言模型（LLM）灌输了拒绝恶意请求的关键能力。之前的作品将这种拒绝机制建模为激活空间中的单一线性方向。我们认为这是一种过于简单化的做法，将两个功能上不同的神经过程混为一谈：伤害的检测和拒绝的执行。在这项工作中，我们将这个单一的表示解构为伤害检测方向和拒绝执行方向。利用这个细粒度模型，我们引入了差异双向干预（DBDI），这是一种新的白盒框架，可以精确地中和关键层的安全对齐。DBDI对拒绝执行方向应用自适应投影无效，同时通过直接转向抑制伤害检测方向。大量实验表明，DBDI优于著名的越狱方法，对Llama-2等模型的攻击成功率高达97.88%。通过提供更细粒度和机械化的框架，我们的工作为深入了解LLM安全对齐提供了新的方向。



## **43. iSeal: Encrypted Fingerprinting for Reliable LLM Ownership Verification**

iSeal：加密指纹识别，实现可靠的LLM所有权验证 cs.CR

Accepted by AAAI 2026

**SubmitDate**: 2025-11-13    [abs](http://arxiv.org/abs/2511.08905v1) [paper-pdf](None)

**Authors**: Zixun Xiong, Gaoyi Wu, Qingyang Yu, Mingyu Derek Ma, Lingfeng Yao, Miao Pan, Xiaojiang Du, Hao Wang

**Abstract**: Given the high cost of large language model (LLM) training from scratch, safeguarding LLM intellectual property (IP) has become increasingly crucial. As the standard paradigm for IP ownership verification, LLM fingerprinting thus plays a vital role in addressing this challenge. Existing LLM fingerprinting methods verify ownership by extracting or injecting model-specific features. However, they overlook potential attacks during the verification process, leaving them ineffective when the model thief fully controls the LLM's inference process. In such settings, attackers may share prompt-response pairs to enable fingerprint unlearning or manipulate outputs to evade exact-match verification. We propose iSeal, the first fingerprinting method designed for reliable verification when the model thief controls the suspected LLM in an end-to-end manner. It injects unique features into both the model and an external module, reinforced by an error-correction mechanism and a similarity-based verification strategy. These components are resistant to verification-time attacks, including collusion-based fingerprint unlearning and response manipulation, backed by both theoretical analysis and empirical results. iSeal achieves 100 percent Fingerprint Success Rate (FSR) on 12 LLMs against more than 10 attacks, while baselines fail under unlearning and response manipulations.

摘要: 鉴于大型语言模型（LLM）从头开始培训的高成本，保护LLM知识产权（IP）变得越来越重要。因此，作为IP所有权验证的标准范式，LLM指纹识别在应对这一挑战方面发挥着至关重要的作用。现有的LLM指纹识别方法通过提取或注入特定于模型的特征来验证所有权。然而，它们在验证过程中忽视了潜在的攻击，从而在模型窃贼完全控制LLM的推理过程时使它们无效。在此类设置中，攻击者可能会共享预算响应对，以启用指纹取消学习或操纵输出以逃避精确匹配验证。我们提出了iSeal，这是第一种指纹识别方法，旨在当模型窃贼以端到端的方式控制可疑的LLM时进行可靠验证。它将独特的功能注入到模型和外部模块中，并通过错误纠正机制和基于相似性的验证策略来加强。这些组件能够抵抗验证时攻击，包括基于共谋的指纹取消学习和响应操纵，并得到理论分析和经验结果的支持。iSeal在12个LLM上针对10多种攻击实现了100%的指纹成功率（FSR），而基线在取消学习和响应操纵下失败。



## **44. ConfGuard: A Simple and Effective Backdoor Detection for Large Language Models**

ConfGuard：大型语言模型简单有效的后门检测 cs.CR

This is an extended version of the copyrighted publication at AAAI

**SubmitDate**: 2025-11-12    [abs](http://arxiv.org/abs/2508.01365v3) [paper-pdf](None)

**Authors**: Zihan Wang, Rui Zhang, Hongwei Li, Wenshu Fan, Wenbo Jiang, Qingchuan Zhao, Guowen Xu

**Abstract**: Backdoor attacks pose a significant threat to Large Language Models (LLMs), where adversaries can embed hidden triggers to manipulate LLM's outputs. Most existing defense methods, primarily designed for classification tasks, are ineffective against the autoregressive nature and vast output space of LLMs, thereby suffering from poor performance and high latency. To address these limitations, we investigate the behavioral discrepancies between benign and backdoored LLMs in output space. We identify a critical phenomenon which we term sequence lock: a backdoored model generates the target sequence with abnormally high and consistent confidence compared to benign generation. Building on this insight, we propose ConfGuard, a lightweight and effective detection method that monitors a sliding window of token confidences to identify sequence lock. Extensive experiments demonstrate ConfGuard achieves a near 100\% true positive rate (TPR) and a negligible false positive rate (FPR) in the vast majority of cases. Crucially, the ConfGuard enables real-time detection almost without additional latency, making it a practical backdoor defense for real-world LLM deployments.

摘要: 后门攻击对大型语言模型（LLM）构成重大威胁，对手可以嵌入隐藏触发器来操纵LLM的输出。大多数现有的防御方法主要是为分类任务设计的，对LLM的自回归性质和巨大的输出空间无效，从而遭受性能差和延迟高的影响。为了解决这些限制，我们调查了输出空间中良性和后门LLM之间的行为差异。我们发现了一个关键现象，我们称之为序列锁：与良性生成相比，后门模型以异常高且一致的置信度生成目标序列。基于这一见解，我们提出了ConfGuard，这是一种轻量级且有效的检测方法，可以监控令牌置信度的滑动窗口以识别序列锁。大量实验表明，在绝大多数情况下，ConfGuard的真阳性率（TPA）接近100%，假阳性率（FPR）可忽略不计。至关重要的是，ConfGuard几乎无需额外延迟即可实现实时检测，使其成为现实世界LLM部署的实用后门防御。



## **45. MCP-RiskCue: Can LLM Infer Risk Information From MCP Server System Logs?**

MCP-RiskCue：LLM可以从HCP服务器系统收件箱推断风险信息吗？ cs.CR

**SubmitDate**: 2025-11-13    [abs](http://arxiv.org/abs/2511.05867v2) [paper-pdf](None)

**Authors**: Jiayi Fu, Qiyao Sun

**Abstract**: Large language models (LLMs) demonstrate strong capabilities in solving complex tasks when integrated with external tools. The Model Context Protocol (MCP) has become a standard interface for enabling such tool-based interactions. However, these interactions introduce substantial security concerns, particularly when the MCP server is compromised or untrustworthy. While prior benchmarks primarily focus on prompt injection attacks or analyze the vulnerabilities of LLM MCP interaction trajectories, limited attention has been given to the underlying system logs associated with malicious MCP servers. To address this gap, we present the first synthetic benchmark for evaluating LLMs ability to identify security risks from system logs. We define nine categories of MCP server risks and generate 1,800 synthetic system logs using ten state-of-the-art LLMs. These logs are embedded in the return values of 243 curated MCP servers, yielding a dataset of 2,421 chat histories for training and 471 queries for evaluation. Our pilot experiments reveal that smaller models often fail to detect risky system logs, leading to high false negatives. While models trained with supervised fine-tuning (SFT) tend to over-flag benign logs, resulting in elevated false positives, Reinforcement Learning from Verifiable Reward (RLVR) offers a better precision-recall balance. In particular, after training with Group Relative Policy Optimization (GRPO), Llama3.1-8B-Instruct achieves 83% accuracy, surpassing the best-performing large remote model by 9 percentage points. Fine-grained, per-category analysis further underscores the effectiveness of reinforcement learning in enhancing LLM safety within the MCP framework. Code and data are available at: https://github.com/PorUna-byte/MCP-RiskCue

摘要: 大型语言模型（LLM）在与外部工具集成时表现出解决复杂任务的强大能力。模型上下文协议（HCP）已成为实现此类基于工具的交互的标准界面。然而，这些交互会带来巨大的安全问题，特别是当HCP服务器受到损害或不值得信任时。虽然之前的基准测试主要关注即时注入攻击或分析LLM LCP交互轨迹的漏洞，但对与恶意LCP服务器相关的底层系统日志的关注有限。为了解决这一差距，我们提出了第一个合成基准，用于评估LLM从系统日志中识别安全风险的能力。我们定义了九种类别的LCP服务器风险，并使用十个最先进的LLM生成1，800个合成系统日志。这些日志嵌入到243个精心策划的LCP服务器的返回值中，生成包含2，421个用于训练的聊天历史和471个用于评估的查询的数据集。我们的试点实验表明，较小的模型通常无法检测到有风险的系统日志，从而导致高假阴性。虽然用监督式微调（SFT）训练的模型往往会过度标记良性日志，导致误报率升高，但来自可验证奖励的强化学习（WLVR）提供了更好的精确度-召回平衡。特别是，经过组相对策略优化（GRPO）训练后，Llama3.1- 8B-Direct的准确率达到了83%，超过性能最好的大型远程模型9个百分点。细粒度、按类别分析进一步强调了强化学习在增强LCP框架内LLM安全性方面的有效性。代码和数据可访问：https://github.com/PorUna-byte/MCP-RiskCue



## **46. Prompt Injection as an Emerging Threat: Evaluating the Resilience of Large Language Models**

提示注入作为一种新兴威胁：评估大型语言模型的弹性 cs.CR

10 pages, 6 figures

**SubmitDate**: 2025-11-13    [abs](http://arxiv.org/abs/2511.01634v2) [paper-pdf](None)

**Authors**: Daniyal Ganiuly, Assel Smaiyl

**Abstract**: Large Language Models (LLMs) are increasingly used in intelligent systems that perform reasoning, summarization, and code generation. Their ability to follow natural-language instructions, while powerful, also makes them vulnerable to a new class of attacks known as prompt injection. In these attacks, hidden or malicious instructions are inserted into user inputs or external content, causing the model to ignore its intended task or produce unsafe responses. This study proposes a unified framework for evaluating how resistant Large Language Models (LLMs) are to prompt injection attacks. The framework defines three complementary metrics such as the Resilience Degradation Index (RDI), Safety Compliance Coefficient (SCC), and Instructional Integrity Metric (IIM) to jointly measure robustness, safety, and semantic stability. We evaluated four instruction-tuned models (GPT-4, GPT-4o, LLaMA-3 8B Instruct, and Flan-T5-Large) on five common language tasks: question answering, summarization, translation, reasoning, and code generation. Results show that GPT-4 performs best overall, while open-weight models remain more vulnerable. The findings highlight that strong alignment and safety tuning are more important for resilience than model size alone. Results show that all models remain partially vulnerable, especially to indirect and direct-override attacks. GPT-4 achieved the best overall resilience (RDR = 9.8 %, SCR = 96.4 %), while open-source models exhibited higher performance degradation and lower safety scores. The findings demonstrate that alignment strength and safety tuning play a greater role in resilience than model size alone. The proposed framework offers a structured, reproducible approach for assessing model robustness and provides practical insights for improving LLM safety and reliability.

摘要: 大型语言模型（LLM）越来越多地用于执行推理、总结和代码生成的智能系统中。它们遵循自然语言指令的能力虽然强大，但也使它们容易受到称为提示注入的一类新型攻击。在这些攻击中，隐藏或恶意指令被插入到用户输入或外部内容中，导致模型忽略其预期任务或产生不安全的响应。这项研究提出了一个统一的框架来评估大型语言模型（LLM）对引发注入攻击的抵抗力。该框架定义了三个补充指标，例如弹性退化指数（RDI）、安全合规系数（SCC）和指令完整性指标（IIM），以联合衡量稳健性、安全性和语义稳定性。我们评估了四种经过翻译调整的模型（GPT-4、GPT-4 o、LLaMA-3 8B Direcct和Flan-T5-Large），用于五种常见语言任务：问题回答、总结、翻译、推理和代码生成。结果显示，GPT-4总体表现最好，而开重模型仍然更脆弱。研究结果强调，强对齐和安全调整对于弹性来说比模型尺寸更重要。结果表明，所有模型仍然部分容易受到攻击，尤其是受到间接和直接覆盖攻击的影响。GPT-4实现了最好的整体弹性（RDR = 9.8%，SCP = 96.4%），而开源模型表现出更高的性能退化和更低的安全评分。研究结果表明，对齐强度和安全调整比模型尺寸本身在弹性方面发挥更大的作用。提出的框架提供了一种结构化、可重复的方法来评估模型稳健性，并为提高LLM安全性和可靠性提供了实用见解。



## **47. Unlearning Imperative: Securing Trustworthy and Responsible LLMs through Engineered Forgetting**

放弃学习势在必行：通过精心设计的遗忘确保值得信赖和负责任的LLM cs.LG

14 pages, 4 figures, 4 tables

**SubmitDate**: **NEW** 2025-11-14    [abs](http://arxiv.org/abs/2511.09855v1) [paper-pdf](None)

**Authors**: James Jin Kang, Dang Bui, Thanh Pham, Huo-Chong Ling

**Abstract**: The growing use of large language models in sensitive domains has exposed a critical weakness: the inability to ensure that private information can be permanently forgotten. Yet these systems still lack reliable mechanisms to guarantee that sensitive information can be permanently removed once it has been used. Retraining from the beginning is prohibitively costly, and existing unlearning methods remain fragmented, difficult to verify, and often vulnerable to recovery. This paper surveys recent research on machine unlearning for LLMs and considers how far current approaches can address these challenges. We review methods for evaluating whether forgetting has occurred, the resilience of unlearned models against adversarial attacks, and mechanisms that can support user trust when model complexity or proprietary limits restrict transparency. Technical solutions such as differential privacy, homomorphic encryption, federated learning, and ephemeral memory are examined alongside institutional safeguards including auditing practices and regulatory frameworks. The review finds steady progress, but robust and verifiable unlearning is still unresolved. Efficient techniques that avoid costly retraining, stronger defenses against adversarial recovery, and governance structures that reinforce accountability are needed if LLMs are to be deployed safely in sensitive applications. By integrating technical and organizational perspectives, this study outlines a pathway toward AI systems that can be required to forget, while maintaining both privacy and public trust.

摘要: 大型语言模型在敏感领域的使用越来越多，暴露了一个关键弱点：无法确保私人信息可以被永久遗忘。然而，这些系统仍然缺乏可靠的机制来保证敏感信息一旦被使用就可以被永久删除。从一开始的再培训成本高得令人望而却步，而且现有的学习方法仍然支离破碎、难以验证，并且往往很容易恢复。本文调查了最近关于LLM机器去学习的研究，并考虑了当前方法可以在多大程度上解决这些挑战。我们回顾了评估是否发生遗忘的方法、未学习的模型对抗对抗攻击的弹性，以及在模型复杂性或专有限制限制透明度时可以支持用户信任的机制。与审计实践和监管框架等机构保障措施一起审查了差异隐私、同质加密、联邦学习和短暂记忆等技术解决方案。审查发现取得了稳步进展，但稳健且可验证的取消学习仍未得到解决。如果要在敏感应用程序中安全部署LLM，就需要避免昂贵的再培训的高效技术、针对对抗性恢复的更强防御以及加强问责制的治理结构。通过整合技术和组织角度，这项研究概述了一条通往人工智能系统的途径，这些系统可能被要求忘记，同时维护隐私和公众信任。



## **48. Biologically-Informed Hybrid Membership Inference Attacks on Generative Genomic Models**

对生成性基因组模型的生物知情混合成员推断攻击 cs.CR

**SubmitDate**: **NEW** 2025-11-14    [abs](http://arxiv.org/abs/2511.07503v2) [paper-pdf](None)

**Authors**: Asia Belfiore, Jonathan Passerat-Palmbach, Dmitrii Usynin

**Abstract**: The increased availability of genetic data has transformed genomics research, but raised many privacy concerns regarding its handling due to its sensitive nature. This work explores the use of language models (LMs) for the generation of synthetic genetic mutation profiles, leveraging differential privacy (DP) for the protection of sensitive genetic data. We empirically evaluate the privacy guarantees of our DP modes by introducing a novel Biologically-Informed Hybrid Membership Inference Attack (biHMIA), which combines traditional black box MIA with contextual genomics metrics for enhanced attack power. Our experiments show that both small and large transformer GPT-like models are viable synthetic variant generators for small-scale genomics, and that our hybrid attack leads, on average, to higher adversarial success compared to traditional metric-based MIAs.

摘要: 遗传数据可用性的增加改变了基因组学研究，但由于其敏感性，对其处理提出了许多隐私问题。这项工作探索了使用语言模型（LM）来生成合成基因突变谱，利用差异隐私（DP）来保护敏感遗传数据。我们通过引入一种新型的生物知情混合成员推断攻击（biHMIA）来经验性地评估DP模式的隐私保证，该攻击将传统的黑匣子MIA与上下文基因组学指标相结合，以增强攻击能力。我们的实验表明，小型和大型Transformer GPT类模型都是小规模基因组学的可行合成变体生成器，并且与传统的基于度量的MIA相比，我们的混合攻击平均会导致更高的对抗成功。



## **49. From Capabilities to Performance: Evaluating Key Functional Properties of LLM Architectures in Penetration Testing**

从能力到性能：在渗透测试中评估LLM架构的关键功能属性 cs.AI

**SubmitDate**: **NEW** 2025-11-14    [abs](http://arxiv.org/abs/2509.14289v3) [paper-pdf](None)

**Authors**: Lanxiao Huang, Daksh Dave, Tyler Cody, Peter Beling, Ming Jin

**Abstract**: Large language models (LLMs) are increasingly used to automate or augment penetration testing, but their effectiveness and reliability across attack phases remain unclear. We present a comprehensive evaluation of multiple LLM-based agents, from single-agent to modular designs, across realistic penetration testing scenarios, measuring empirical performance and recurring failure patterns. We also isolate the impact of five core functional capabilities via targeted augmentations: Global Context Memory (GCM), Inter-Agent Messaging (IAM), Context-Conditioned Invocation (CCI), Adaptive Planning (AP), and Real-Time Monitoring (RTM). These interventions support, respectively: (i) context coherence and retention, (ii) inter-component coordination and state management, (iii) tool use accuracy and selective execution, (iv) multi-step strategic planning, error detection, and recovery, and (v) real-time dynamic responsiveness. Our results show that while some architectures natively exhibit subsets of these properties, targeted augmentations substantially improve modular agent performance, especially in complex, multi-step, and real-time penetration testing tasks.

摘要: 大型语言模型（LLM）越来越多地用于自动化或增强渗透测试，但它们在攻击阶段的有效性和可靠性仍不清楚。我们在现实的渗透测试场景中对多个基于LLM的代理（从单代理到模块化设计）进行了全面评估，测量经验性能和反复出现的故障模式。我们还通过有针对性的增强来隔离五种核心功能能力的影响：全球上下文记忆（GCM）、代理间消息传递（ILM）、上下文条件调用（CI）、自适应规划（AP）和实时监控（RTI）。这些干预措施分别支持：（i）上下文一致性和保留，（ii）组件间协调和状态管理，（iii）工具使用准确性和选择性执行，（iv）多步骤战略规划、错误检测和恢复，以及（v）实时动态响应能力。我们的结果表明，虽然一些架构本身表现出这些属性的子集，但有针对性的增强可以大大提高模块化代理的性能，特别是在复杂、多步骤和实时渗透测试任务中。



## **50. Chain-of-Lure: A Universal Jailbreak Attack Framework using Unconstrained Synthetic Narratives**

Chain-of-Lure：一个使用无约束合成叙述的通用越狱攻击框架 cs.CR

23 pages, 3 main figures

**SubmitDate**: **NEW** 2025-11-14    [abs](http://arxiv.org/abs/2505.17519v2) [paper-pdf](None)

**Authors**: Wenhan Chang, Tianqing Zhu, Yu Zhao, Shuangyong Song, Ping Xiong, Wanlei Zhou

**Abstract**: In the era of rapid generative AI development, interactions with large language models (LLMs) pose increasing risks of misuse. Prior research has primarily focused on attacks using template-based prompts and optimization-oriented methods, while overlooking the fact that LLMs possess strong unconstrained deceptive capabilities to attack other LLMs. This paper introduces a novel jailbreaking method inspired by the Chain-of-Thought mechanism. The attacker employs mission transfer to conceal harmful user intent within dialogue and generates a progressive chain of lure questions without relying on predefined templates, enabling successful jailbreaks. To further improve the attack's strength, we incorporate a helper LLM model that performs randomized narrative optimization over multi-turn interactions, enhancing the attack performance while preserving alignment with the original intent. We also propose a toxicity-based framework using third-party LLMs to evaluate harmful content and its alignment with malicious intent. Extensive experiments demonstrate that our method consistently achieves high attack success rates and elevated toxicity scores across diverse types of LLMs under black-box API settings. These findings reveal the intrinsic potential of LLMs to perform unrestricted attacks in the absence of robust alignment constraints. Our approach offers data-driven insights to inform the design of future alignment mechanisms. Finally, we propose two concrete defense strategies to support the development of safer generative models.

摘要: 在快速生成式人工智能发展的时代，与大型语言模型（LLM）的交互带来了越来越大的滥用风险。之前的研究主要集中在使用基于模板的提示和面向优化的方法的攻击上，而忽视了LLM拥有强大的不受限制的欺骗能力来攻击其他LLM这一事实。本文介绍了一种受思想链机制启发的新颖越狱方法。攻击者利用任务转移来隐藏对话中的有害用户意图，并在不依赖预定义模板的情况下生成渐进的诱饵问题链，从而实现成功越狱。为了进一步提高攻击的强度，我们引入了一个助手LLM模型，该模型在多回合交互中执行随机叙事优化，增强攻击性能，同时保持与最初意图的一致。我们还提出了一个基于毒性的框架，使用第三方LLM来评估有害内容及其与恶意意图的一致性。大量实验表明，在黑匣子API设置下，我们的方法在不同类型的LLM中始终实现了高攻击成功率和更高的毒性评分。这些发现揭示了LLM在缺乏稳健对齐约束的情况下执行无限制攻击的内在潜力。我们的方法提供数据驱动的见解，为未来对齐机制的设计提供信息。最后，我们提出了两种具体的防御策略来支持更安全的生成模型的开发。



