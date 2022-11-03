## Efﬁcient and Optimal LPM Caching with Splicing
Additional code for the Thesis by Itamar Gozlan (M.Sc) <br>
Under the supervision of Dr. Gil Einziger, and Dr. Gabriel Scalosub. <br>
Ben-Gurion University of the Negev <br>
The Faculty of Natural Sciences <br>

## Description
Caching forwarding rules in switches differ from most general caching settings because forwarding rules 
often have an underlying dependencies structure. <br>
In an LPM matching, A packet is matched to the rules according to its header fields (typically the destination IP 
address), and the switch selects the forwarding rule with the longest prefix match.
Therefore, when a rule is admitted to the cache, all other rules which share its prescribed prefix. 
(but are more specific) must also be inserted into the cache to maintain consistency.

Rule splicing [[1]](#1) is a technique that allows admitting a forwarding rule to the cache without forcing all of its 
dependent rules to be included in the cache as well.
Splicing is done by modifying the forwarding rules to divert the traffic corresponding to missing dependent rules 
directly to the controller.


### Project Overview
This project is divided into pre\post processing, driver code, and algorithm implementation.

``Algoritm.py`` - Contain implementation for 4 different algorithms:
1. Optimal cache without splicing
2. Greedy Caching with splicing
3. Optimal Caching with splicing
4. MixedSet ,LPM version implementation [[1]](#1)

``Caida.py`` - Is used to parse and create policy from Caida traces [[2]](#1) <br>
``Utils.py`` - The algorithms uses ``networkx`` graph framework, this file hold helper function that wrap ``networkx``<br>
``Zipf.py`` - Construct traces by the publicly Stanford Backbone Router policy [[3]](#1)<br>
``process_result.py`` - Multiple functions to post-process results.<br>
``simulator_main.py`` - Main driver code to run simulations<br>

### Run algorithms
Usage example:<br>
``python simulator_main.py 1 traces/caida_traceTCP_prefix_weight.json myresult/``

To run the algorithm, you can either generate policies with different weights yourself or use the
traces that were previously computed under the ``traces/`` directory.

The run command is <br>
``python simulator_main.py <ALGORITHM_TYPE> <INPUT_TRACE_JSON> <OUTPUT_DIR>``

The program receives 3 inputs:<br>
``ALGORITHM_TYPE``:0, 1,2,3 such that: <br>
0: Optimal cache without splicing <br>
1: Greedy Caching with splicing <br>
2: Optimal Caching with splicing <br>
3: MixedSet,LPM version implementation <br>

``INPUT_TRACE_JSON`` - JSON file of dictionary of prefix: weight for each rule. 

``OUTPUT_DIR`` - Target dir to save results

### References


<a id="1">[1]</a> Katta, Naga and Alipourfard, Omid and Rexford, Jennifer and Walker, David, 
"Cacheflow: Dependency-aware rule-caching for software-defined networks",
Proceedings of the Symposium on SDN Research, 2016

<a id="2">[2]</a> Caida anonymized Internet Traces 2019, https://catalog.caida.org/details/dataset/passive\_2019\_pcap

<a id="2">[3]</a> Stanford backbone router forwarding configuration, http://tinyurl.com/oaezlha