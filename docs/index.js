{

    /*******************/
    /* Misc. Utilities */
    /*******************/
    
    const isUndefined = value => value === void(0);

    const randomInt = (minimum, maximum) => Math.floor(Math.random() * (maximum - minimum + 1)) + minimum;
    
    const randomChoice = array => array[randomInt(0, array.length-1)];

    const shuffle = array => array.sort(() => Math.random() - 0.5);

    const range = (start, end) => {
        const items = [];
        for (let i = start; i < end; i++) {
            items.push(i);
        }
        return items;
    };

    const zip = rows => rows[0].map((_,c) => rows.map(row => row[c]));

    const lerp = (start, end, interpolationAmount) => start + interpolationAmount * (end - start);
    
    const sum = inputArray => inputArray.reduce((a, b) => a + b, 0);
    
    const mean = inputArray => sum(inputArray) / inputArray.length;
    
    const numberToOrdinal = (number) => {
        const onesDigit = number % 10;
        const tensDigit = number % 100;
        if (onesDigit == 1 && tensDigit != 11) {
            return number + 'st';
        } else if (onesDigit == 2 && tensDigit != 12) {
            return number + 'nd';
        } else if (onesDigit == 3 && tensDigit != 13) {
            return number + 'rd';
        } else {
            return number + 'th';
        }
    };
    
    // D3 Extensions
    d3.selection.prototype.moveToFront = function() {
	return this.each(function() {
	    if (this.parentNode !== null) {
		this.parentNode.appendChild(this);
	    }
	});
    };

    d3.selection.prototype.moveToBack = function() {
        return this.each(function() {
            var firstChild = this.parentNode.firstChild;
            if (firstChild) {
                this.parentNode.insertBefore(this, firstChild);
            }
        });
    };

    /**********************/
    /* HTML Element Utils */
    /**********************/
    
    const removeAllChildNodes = (parent) => {
        while (parent.firstChild) {
            parent.removeChild(parent.firstChild);
        }
    };
    
    const createNewElement = (childTag, {classes, attributes, innerHTML}={}) => {
        const newElement = childTag === 'svg' ? document.createElementNS('http://www.w3.org/2000/svg', childTag) : document.createElement(childTag);
        if (!isUndefined(classes)) {
            classes.forEach(childClass => newElement.classList.add(childClass));
        }
        if (!isUndefined(attributes)) {
            Object.entries(attributes).forEach(([attributeName, attributeValue]) => {
                newElement.setAttribute(attributeName, attributeValue);
            });
        }
        if (!isUndefined(innerHTML)) {
            newElement.innerHTML = innerHTML;
        }
        return newElement;
    };
    
    const createTableWithElements = (rows, {classes, attributes}={}) => {
        const table = createNewElement('table', {classes, attributes});
        rows.forEach(elements => {
            const tr = document.createElement('tr');
            table.append(tr);
            elements.forEach(element => {
                const td = document.createElement('td');
                tr.append(td);
                td.append(element);
            });
        });
        return table;
    };
    
    const createLazyAccordion = (labelGeneratorDestructorTriples) => {
        const accordionContainer = createNewElement('div', {classes: ['accordion-container']});
        labelGeneratorDestructorTriples.forEach(([labelInnerHTML, contentGenerator, contentDestructor], i) => {
            // contentGenerator and contentDestructor take an HTML element
            const labelContentPairElement = createNewElement('div');
            const label = createNewElement('p', {classes: ['accordion-label'], innerHTML: labelInnerHTML});
            const contentDiv = createNewElement('div', {classes: ['accordion-content']});
            labelContentPairElement.append(label);
            labelContentPairElement.append(contentDiv);
            label.onclick = () => {
                label.classList.toggle('active');
                contentDiv.classList.toggle('active');
                if (label.classList.contains('active')) {
                    contentGenerator(contentDiv);
                } else {
                    contentDestructor(contentDiv);
                }
            };
            accordionContainer.append(labelContentPairElement);
        });
        return accordionContainer;
    };
    
    /*********************/
    /* Result Accordions */
    /*********************/
    
    const numberOfResultsToShow = 20;
    
    d3.json('./hyperparameter_search_results.json').then(resultSummaryDictsWithoutIds => { // All Results
        const resultSummaryDicts = resultSummaryDictsWithoutIds.map((resultSummaryDict, i) => Object.assign(resultSummaryDict, {trial_id: i}));
        document.querySelector('#total-number-of-hyperparameter-search-trials-span').innerHTML = resultSummaryDicts.length.toString();
        const maxValidationAccuracy = Math.max(...resultSummaryDicts.map(resultSummaryDict => resultSummaryDict.validation_accuracy));
        const resultDictsWithMaxValidationAccuracy = resultSummaryDicts.filter(resultSummaryDict => resultSummaryDict.validation_accuracy ===  maxValidationAccuracy);
        document.querySelector('#total-number-of-models-with-max-validation-accuracy').innerHTML = resultDictsWithMaxValidationAccuracy.length.toString();
        document.querySelector('#max-validation-accuracy-percent-span').innerHTML = `${(100*maxValidationAccuracy).toFixed(2)}% (${maxValidationAccuracy*19}/19 correct)`; // @todo it would be nice to not hard-code this 19
        const worstTestingAccuracyWithMaxValidationAccuracy = resultDictsWithMaxValidationAccuracy.reduce(
            (accumulator, resultDict) => ((resultDict.testing_accuracy < accumulator) ? resultDict.testing_accuracy : accumulator)
            , 2);
        document.querySelector('#worst-testing-accuracy-with-max-validation-accuracy-percent-span').innerHTML = // @todo it would be nice to not hard-code this 19
            `${(100*worstTestingAccuracyWithMaxValidationAccuracy).toFixed(2)}% (${worstTestingAccuracyWithMaxValidationAccuracy*19}/19 correct)`;
        
        const generateLabelGeneratorDestructorTriple = (resultSummaryDict, rankingCriterionString, i) => {
            
            const labelInnerHTML = `${numberToOrdinal(i+1)} Place Best ${rankingCriterionString} Model (Trial #${resultSummaryDict.trial_id})`;
            
            const contentGenerator = contentContainer => {
                contentContainer.append(createNewElement('p', {innerHTML: 'Results:', classes: ['result-table-title']}));
                contentContainer.append(createTableWithElements([
                    [
                        createNewElement('p', {innerHTML: 'Testing Accuracy'}),
                        createNewElement('p', {attributes: {style: 'float: right;'}, innerHTML: `${(100*resultSummaryDict.testing_accuracy).toFixed(2)}%, ${resultSummaryDict.testing_accuracy*19}/19`})
                    ],
                    [
                        createNewElement('p', {innerHTML: 'Validation Accuracy'}),
                        createNewElement('p', {attributes: {style: 'float: right;'}, innerHTML: `${(100*resultSummaryDict.validation_accuracy).toFixed(2)}%, ${resultSummaryDict.validation_accuracy*19}/19`})
                    ],
                    [
                        createNewElement('p', {innerHTML: 'Training Accuracy'}),
                        createNewElement('p', {attributes: {style: 'float: right;'}, innerHTML: `${(100*resultSummaryDict.training_accuracy).toFixed(2)}%, ${resultSummaryDict.training_accuracy*resultSummaryDict.training_set_batch_count*resultSummaryDict.batch_size}/${resultSummaryDict.training_set_batch_count*resultSummaryDict.batch_size}`})
                    ],
                    [
                        createNewElement('p', {innerHTML: 'Total Dataset Accuracy'}),
                        createNewElement('p', {attributes: {style: 'float: right;'}, innerHTML: `${(100*resultSummaryDict.total_accuracy).toFixed(2)}%, ${resultSummaryDict.total_accuracy*188}/188`})
                    ],
                    [
                        createNewElement('p', {innerHTML: 'Validation Loss'}),
                        createNewElement('p', {attributes: {style: 'float: right;'}, innerHTML: `${resultSummaryDict.best_validation_loss}`})
                    ],
                ], {classes: ['result-table', 'container-center'], attributes: {style: 'margin-bottom: 1em;'}}));
                
                contentContainer.append(createNewElement('p', {innerHTML: 'graph2vec Hyperparameters:', classes: ['result-table-title']}));
                contentContainer.append(createTableWithElements([
                    [createNewElement('p', {innerHTML: 'Weisfeiler-Lehman Iteration Count'}), createNewElement('p', {attributes: {style: 'float: right;'}, innerHTML: `${resultSummaryDict.wl_iterations}`})],
                    [createNewElement('p', {innerHTML: 'Embedding Size'}), createNewElement('p', {attributes: {style: 'float: right;'}, innerHTML: `${resultSummaryDict.embedding_size}`})],
                    [createNewElement('p', {innerHTML: 'Number of graph2vec Training Epochs'}), createNewElement('p', {attributes: {style: 'float: right;'}, innerHTML: `${resultSummaryDict.graph2vec_epochs}`})],
                    [createNewElement('p', {innerHTML: 'graph2vec Learning Rate'}), createNewElement('p', {attributes: {style: 'float: right;'}, innerHTML: ` ${resultSummaryDict.graph2vec_learning_rate}`})],
                ], {classes: ['result-table', 'container-center'], attributes: {style: 'margin-bottom: 1em;'}}));
                
                contentContainer.append(createNewElement('p', {innerHTML: 'Neural Network Hyperparameters:', classes: ['result-table-title']}));
                contentContainer.append(createTableWithElements([
                    [createNewElement('p', {innerHTML: 'Batch Size'}), createNewElement('p', {attributes: {style: 'float: right;'}, innerHTML: ` ${resultSummaryDict.batch_size}`})],
                    [createNewElement('p', {innerHTML: 'Neural Learning Rate'}), createNewElement('p', {attributes: {style: 'float: right;'}, innerHTML: ` ${resultSummaryDict.classifier_learning_rate}`})],
                    [createNewElement('p', {innerHTML: 'Number of Dense Layers'}), createNewElement('p', {attributes: {style: 'float: right;'}, innerHTML: ` ${resultSummaryDict.number_of_layers}`})],
                    [createNewElement('p', {innerHTML: 'Gradient Clipping Threshold'}), createNewElement('p', {attributes: {style: 'float: right;'}, innerHTML: ` ${resultSummaryDict.gradient_clip_val}`})],
                    [createNewElement('p', {innerHTML: 'Dropout Probability'}), createNewElement('p', {attributes: {style: 'float: right;'}, innerHTML: ` ${resultSummaryDict.dropout_probability}`})],
                ], {classes: ['result-table', 'container-center'], attributes: {style: 'margin-bottom: 1em;'}}));
            };
            
            const contentDestructor = contentContainer => {
                removeAllChildNodes(contentContainer);
            };

            return [labelInnerHTML, contentGenerator, contentDestructor];
        };
        
        const validationAccuracyLabelGeneratorDestructorTriples = resultSummaryDicts
              .sort((resultSummaryDict1, resultSummaryDict2) => {
                  let result = resultSummaryDict2.validation_accuracy-resultSummaryDict1.validation_accuracy;
                  if (result === 0) {
                      result = resultSummaryDict2.testing_accuracy-resultSummaryDict1.testing_accuracy;
                  }
                  return result;
              })
              .slice(0, numberOfResultsToShow)
              .map((resultSummaryDict, i) => generateLabelGeneratorDestructorTriple(resultSummaryDict, 'Validation Accuracy', i));
        const allResultsByValidationAccuracyDiv = document.querySelector('#all-results-by-validation-accuracy');
        const validationaccuracyAccordion = createLazyAccordion(validationAccuracyLabelGeneratorDestructorTriples);
        validationaccuracyAccordion.classList.add('container-center');
        allResultsByValidationAccuracyDiv.append(validationaccuracyAccordion);

        const validationLossLabelGeneratorDestructorTriples = resultSummaryDicts
              .sort((resultSummaryDict1, resultSummaryDict2) => resultSummaryDict1.best_validation_loss-resultSummaryDict2.best_validation_loss)
              .slice(0, numberOfResultsToShow)
              .map((resultSummaryDict, i) => generateLabelGeneratorDestructorTriple(resultSummaryDict, 'Validation Loss', i));
        const allResultsByValidationLossDiv = document.querySelector('#all-results-by-validation-loss');
        const validationLossAccordion = createLazyAccordion(validationLossLabelGeneratorDestructorTriples);
        validationLossAccordion.classList.add('container-center');
        allResultsByValidationLossDiv.append(validationLossAccordion);
    });

    /***************************/
    /* Visualization Utilities */
    /***************************/
    
    const innerMargin = 150;
    const textMargin = 8;
    const curvedArrowOffset = 30;

    const xCenterPositionForIndex = (encompassingSvg, index, total, overridingInnerMargin=innerMargin) => {
        const svgWidth = parseFloat(encompassingSvg.style('width'));
        const innerWidth = svgWidth - 2 * overridingInnerMargin;
        const delta = innerWidth / (total - 1);
        const centerX = overridingInnerMargin + index * delta;
        return centerX;
    };

    const generateTextWithBoundingBox = (encompassingSvg, parentGroupClass, textElementClass, boundingBoxClass, textCenterX, yPosition, textString) => {
        const parentGroup = encompassingSvg
              .append('g')
              .classed(parentGroupClass, true);
        const textElement = parentGroup
              .append('text')
	      .attr('y', yPosition)
              .classed(textElementClass, true)
              .html(textString);
        textElement
	    .attr('x', textCenterX - textElement.node().getBBox().width / 2);
        const boundingBoxElement = parentGroup
              .append('rect')
              .classed(boundingBoxClass, true)
              .attr('x', textElement.attr('x') - textMargin)
              .attr('y', () => {
                  const textElementBBox = textElement.node().getBBox();
                  return textElementBBox.y - textMargin;
              })
              .attr('width', textElement.node().getBBox().width + 2 * textMargin)
              .attr('height', textElement.node().getBBox().height + 2 * textMargin);
        textElement.moveToFront();
        return parentGroup;
    };

    const getD3HandleTopXY = (element) => {
        /* element is a D3 handle */
        const boundingBox = element.node().getBBox();
        const x = boundingBox.x + boundingBox.width/2;
        const y = boundingBox.y;
        return [x, y];
    };

    const getD3HandleBottomXY = (element) => {
        /* element is a D3 handle */
        const boundingBox = element.node().getBBox();
        const x = boundingBox.x + boundingBox.width/2;
        const y = boundingBox.y + boundingBox.height;
        return [x, y];
    };

    const getD3HandleTopLeftXY = (element) => {
        /* element is a D3 handle */
        const boundingBox = element.node().getBBox();
        const x = boundingBox.x;
        const y = boundingBox.y;
        return [x, y];
    };

    const getD3HandleBottomLeftXY = (element) => {
        /* element is a D3 handle */
        const boundingBox = element.node().getBBox();
        const x = boundingBox.x;
        const y = boundingBox.y + boundingBox.height;
        return [x, y];
    };

    const getD3HandleTopRightXY = (element) => {
        /* element is a D3 handle */
        const boundingBox = element.node().getBBox();
        const x = boundingBox.x + boundingBox.width;
        const y = boundingBox.y;
        return [x, y];
    };

    const getD3HandleBottomRightXY = (element) => {
        /* element is a D3 handle */
        const boundingBox = element.node().getBBox();
        const x = boundingBox.x + boundingBox.width;
        const y = boundingBox.y + boundingBox.height;
        return [x, y];
    };

    const defineArrowHead = (encompassingSvg) => {
	const defs = encompassingSvg.append('defs');
	const marker = defs.append('marker')
	      .attr('markerWidth', '10')
	      .attr('markerHeight', '10')
	      .attr('refX', '5')
	      .attr('refY', '3')
	      .attr('orient', 'auto')
	      .attr('id', 'arrowhead');
        const polygon = marker.append('polygon')
	      .attr('points', '0 0, 6 3, 0 6');
    };
    
    
    const drawArrow = (encompassingSvg, [x1, y1], [x2, y2]) => {
        const line = encompassingSvg
              .append('line')
	      .attr('marker-end','url(#arrowhead)')
              .moveToBack()
	      .attr('x1', x1)
	      .attr('y1', y1)
	      .attr('x2', x2)
	      .attr('y2', y2)
              .classed('arrow-line', true);
    };
    
    const drawCurvedArrow = (encompassingSvg, [x1, y1], [x2, y2]) => {
	const midpointX = (x1+x2)/2;
	const midpointY = (y1+y2)/2;
	const dx = (x2 - x1);
	const dy = (y2 - y1);
	const normalization = Math.sqrt((dx * dx) + (dy * dy));
	const offSetX = midpointX + curvedArrowOffset*(dy/normalization);
	const offSetY = midpointY - curvedArrowOffset*(dx/normalization);
	const path = `M ${x1}, ${y1} S ${offSetX}, ${offSetY} ${x2}, ${y2}`;
        const line = encompassingSvg
              .append('path')
	      .attr('marker-end','url(#arrowhead)')
              .moveToBack()
	      .attr('d', path)
              .classed('arrow-line', true);
    };
    
    /******************/
    /* Visualizations */
    /******************/
    
    const appendGraph = (containingGroup, linkData, nodeData, nodeIdToNeighborNodeIds, nodeIdToNode) => {
        
        const edgeGroup = containingGroup
              .append('g')
              .classed('graph-edge-group', true);
        const edgeEnterSelection = edgeGroup
	      .selectAll('line')
	      .data(linkData)
	      .enter()
              .append('line')
              .classed('graph-edge', true);
        
        const nodeGroup = containingGroup
              .append('g')
              .classed('graph-node-group', true);
	const nodeEnterSelection = nodeGroup
	      .selectAll('circle')
	      .data(nodeData)
	      .enter()
              .append('circle')
              .classed('graph-node', true);
        
        const alphaDecay = 0.0075;
        const velocityDecay = 0.8;
        const paddingBetweenNodes = 25;
        const drag = d3.drag();
        drag.on('drag', (datum, i) => {
	    simulation
		.alpha(0.1)
		.restart();
            datum.x += d3.event.dx;
            datum.y += d3.event.dy;
        });
        const simulation = d3.forceSimulation()
	      .alphaDecay(alphaDecay)
	      .velocityDecay(velocityDecay);
	simulation
            .force('center', d3.forceCenter( parseFloat(containingGroup.attr('x')) , parseFloat(containingGroup.attr('y')) + parseFloat(containingGroup.attr('height')) / 2 ))
            .force('collide', d3.forceCollide(paddingBetweenNodes).strength(0.25).iterations(200))
            .force('links', () => {
                const linkForceAlpha = (nodeData.length / linkData.length) * 0.5 * simulation.alpha() + (1 - simulation.alpha()) * 0.0005;
	        nodeData.forEach(node => {
                    const neighborIds = nodeIdToNeighborNodeIds[node.id];
                    const neighbors = [...neighborIds].map(id => nodeIdToNode[id]);
                    const neighborMeanX = mean(neighbors.map(neighbor => neighbor.x));
                    const neighborMeanY = mean(neighbors.map(neighbor => neighbor.y));
                    const distanceToNeighborMean = Math.sqrt((node.x-neighborMeanX)**2 + (node.y-neighborMeanY)**2);
                    node.x = node.x * (1-linkForceAlpha) + linkForceAlpha * neighborMeanX;
                    node.y = node.y * (1-linkForceAlpha) + linkForceAlpha * neighborMeanY;
                });
            })
	    .nodes(nodeData).on('tick', () => {
	    	nodeEnterSelection
	    	    .attr('cx', datum => datum.x)
	    	    .attr('cy', datum => datum.y)
                    .call(drag);
	    	edgeEnterSelection
	    	    .attr('x1', datum => nodeIdToNode[datum.source].x)
	    	    .attr('y1', datum => nodeIdToNode[datum.source].y)
	    	    .attr('x2', datum => nodeIdToNode[datum.target].x)
	    	    .attr('y2', datum => nodeIdToNode[datum.target].y);
	    })
	    .restart();
    };
    
    const generateErdosRenyiGraphData = (nodeCount, probability) => {
        const nodes = range(0, nodeCount).map(i => ({id: i.toString()}));
        const edges = [];
        nodes.forEach(startNode => {
            nodes.forEach(endNode => {
                if (Math.random() < probability) {
                    edges.push({source: startNode.id, target: endNode.id});
                }
            });
        });
        return [nodes, edges];
    };
    
    const appendRandomConnectedGraph = (containingGroup) => {
        const [originalNodeData, originalLinkData] = generateErdosRenyiGraphData(40, 0.01);
        const nodeIdToNeighborNodeIds = originalLinkData.reduce((accumulator, link) => {
            if (!accumulator.hasOwnProperty(link.source)) {
                accumulator[link.source] = [];
            }
            if (!accumulator.hasOwnProperty(link.target)) {
                accumulator[link.target] = [];
            }
            accumulator[link.source].push(link.target);
            accumulator[link.target].push(link.source);
            return accumulator;
        }, {});
        {
            const connectedNodeIds = new Set();
            let lastSeenNodeId = originalLinkData[0].source;
            Object.keys(nodeIdToNeighborNodeIds).forEach(nodeId => {
                if (!connectedNodeIds.has(nodeId)) {
                    nodeIdToNeighborNodeIds[nodeId].push(lastSeenNodeId);
                    nodeIdToNeighborNodeIds[lastSeenNodeId].push(nodeId);
                    connectedNodeIds.add(nodeId);
                    const neighborIds = nodeIdToNeighborNodeIds[nodeId];
                    neighborIds.forEach(neighborId => connectedNodeIds.add(neighborId));
                    lastSeenNodeId = randomChoice(Array.from(connectedNodeIds));
                }
            });
        }
        const linkData = Object.keys(nodeIdToNeighborNodeIds).reduce((accumulator, nodeId) => {
            const neighborIds = nodeIdToNeighborNodeIds[nodeId];
            neighborIds.forEach(neighborId => {
                accumulator.push({source: nodeId, target: neighborId});
            });
            return accumulator;
        }, []);
        const nodeData = Array.from(linkData.reduce((accumulator, link) => {
            accumulator.add(link.source);
            accumulator.add(link.target);
            return accumulator;
        }, new Set())).map(nodeId => ({id: nodeId, x: Math.random()*window.innerWidth*2, y: Math.random()*window.innerHeight*2}));
        const nodeIdToNode = nodeData.reduce((accumulator, node) => {
            accumulator[node.id] = node;
            return accumulator;
        }, {});
        appendGraph(containingGroup, linkData, nodeData, nodeIdToNeighborNodeIds, nodeIdToNode);
    };
    
    const renderNeuralClassifierArchitecture = () => {

        /* Init */
        
        const denseLayerCount = randomInt(2, 5);
        const denseLayerVectorLength = randomInt(4,6);
        const svg = d3.select('#neural-classifier-depiction');
        svg.selectAll('*').remove();
        svg.attr('width', `80vw`);
        defineArrowHead(svg);
        const svgWidth = parseFloat(svg.style('width'));

        /* Blocks */
        
        // Graph
        const graphCenterX = svgWidth/2;
        const graphGroup = svg.append('g')
              .attr('width', 600)
              .attr('height', 400);
        const graphBackgroundRect = graphGroup.append('rect');
        [graphGroup, graphBackgroundRect].forEach(
            d3Handle => d3Handle
                .attr('width', 600)
                .attr('height', 400)
                .attr('y', 100)
                .attr('x', graphCenterX-d3Handle.node().getBBox().width/2)
                .classed('graph-group-background', true)
        );
        graphGroup.classed('graph-group', true);
        appendRandomConnectedGraph(graphGroup);

        // graph2vec Embedding Layer
        const graph2vecCenterX = svgWidth/2;
        const graph2vecGroup = generateTextWithBoundingBox(svg, 'text-with-bbox-group', 'text-with-bbox-group-text', 'text-with-bbox-group-bounding-box', graph2vecCenterX, 600, 'graph2vec Embedding Layer');
        graph2vecGroup.classed('graph2vec-group', true);
        graph2vecGroup.select('rect')
            .attr('x', innerMargin)
            .attr('width', svgWidth - innerMargin*2);
        
        // Dense Layer
        const denseGroups = [];
        range(0, denseLayerCount).forEach(i => {
            const denseCenterX = svgWidth/2;
            const denseGroup = generateTextWithBoundingBox(svg, 'text-with-bbox-group', 'text-with-bbox-group-text', 'text-with-bbox-group-bounding-box', denseCenterX, 700+i*100, 'Dense Layer');
            denseGroup.classed('dense-group', true);
            denseGroup.select('rect')
                .attr('x', innerMargin)
                .attr('width', svgWidth - innerMargin*2);
            denseGroups.push(denseGroup);
        });

        // Fully Connected Layer
        const fullyConnectedCenterX = svgWidth/2;
        const fullyConnectedGroup = generateTextWithBoundingBox(svg, 'text-with-bbox-group', 'text-with-bbox-group-text', 'text-with-bbox-group-bounding-box', fullyConnectedCenterX, 700+denseLayerCount*100, 'Fully Connected Layer');
        fullyConnectedGroup.classed('fully-connected-group', true);
        fullyConnectedGroup.select('rect')
            .attr('x', innerMargin)
            .attr('width', svgWidth - innerMargin*2);

        // Sigmoid Layer
        const sigmoidCenterX = svgWidth/2;
        const sigmoidGroup = generateTextWithBoundingBox(svg, 'text-with-bbox-group', 'text-with-bbox-group-text', 'text-with-bbox-group-bounding-box', sigmoidCenterX, 800+denseLayerCount*100, 'Sigmoid');
        sigmoidGroup.classed('sigmoid-group', true);
        
        // Round Layer
        const roundCenterX = svgWidth/2;
        const roundGroup = generateTextWithBoundingBox(svg, 'text-with-bbox-group', 'text-with-bbox-group-text', 'text-with-bbox-group-bounding-box', roundCenterX, 900+denseLayerCount*100, 'Round');
        roundGroup.classed('round-group', true);
        
        /* Arrows */

        // Graph to graph2vec Embedding Layer Arrows
        drawArrow(svg, getD3HandleBottomXY(graphBackgroundRect), getD3HandleTopXY(graph2vecGroup));

        // graph2vec Embedding Layer to Dense Layers to Fully Connected Layer Arrows
        const multiArrowGroups = [graph2vecGroup].concat(denseGroups, [fullyConnectedGroup]);
        const xMinMultiArrowGroup = innerMargin;
        const xMaxMultiArrowGroup = svgWidth - innerMargin*2;
        multiArrowGroups.forEach((topGroup, i) => {
            if (i < denseGroups.length+1) {
                const topGroupY = getD3HandleBottomXY(topGroup)[1];
                const bottomGroup = multiArrowGroups[i+1];
                const bottomGroupY = getD3HandleTopXY(bottomGroup)[1];
                range(0, denseLayerVectorLength).forEach(xTopIndex => {
                    const topX = innerMargin/2+lerp(xMinMultiArrowGroup, xMaxMultiArrowGroup, xTopIndex/(denseLayerVectorLength-1));
                    range(0, denseLayerVectorLength).forEach(xBottomIndex => {
                        const bottomX = innerMargin/2+lerp(xMinMultiArrowGroup, xMaxMultiArrowGroup, xBottomIndex/(denseLayerVectorLength-1));
                        drawArrow(svg, [topX, topGroupY], [bottomX, bottomGroupY]);
                    });
                });
            }
        });

        // Fully Connected Layer to Sigmoid Layer Arrows
        drawArrow(svg, getD3HandleBottomXY(fullyConnectedGroup), getD3HandleTopXY(sigmoidGroup));

        // Sigmoid Layer to Round LayerArrows
        drawArrow(svg, getD3HandleBottomXY(sigmoidGroup), getD3HandleTopXY(roundGroup));

        const svgGroupBottomYValues = [];
        svg.selectAll('g').each(function() { svgGroupBottomYValues.push(getD3HandleBottomXY(d3.select(this))[1]); });
        svg.attr('height', `${Math.max(...svgGroupBottomYValues)+100}px`);
        
    };
    renderNeuralClassifierArchitecture();
    let currentWindowWidth = window.innerWidth;
    window.addEventListener('resize', () => {
        if (currentWindowWidth !== window.innerWidth) {
            currentWindowWidth = window.innerWidth;
            renderNeuralClassifierArchitecture();
        }
    });
    
    /*******************************/
    /* Example Graph Visualization */
    /*******************************/

    d3.json('./mutag_data.json').then(mutagData => {
        const svgContainer = document.querySelector('#data-set-example-container');
        const svg = d3.select(svgContainer.querySelector('svg'))
            .attr('width', 600)
            .attr('height', 400);
        const svgBoundingBox = svg.node().getBBox();
        const graphContainer = svg.append('g');
        graphContainer
            .attr('x', 300)
            .attr('y', 0)
            .attr('width', 600)
            .attr('height', 400);
        const exampleId = randomChoice(Object.keys(mutagData));
        document.querySelector('#mutag-example-graph-id').innerHTML = exampleId.toString();
        const exampleGraph = mutagData[exampleId].graph;
        const linkData = exampleGraph.links.reduce((accumulator, nodedDatum) => {
            accumulator.push({'source': nodedDatum.source.toString(), 'target': nodedDatum.target.toString()});
            return accumulator;
        }, []);
        const nodeData = exampleGraph.nodes.reduce((accumulator, nodedDatum) => {
            accumulator.push({'id': nodedDatum.id.toString(), x: Math.random()*window.innerWidth*2, y: Math.random()*window.innerHeight*2});
            return accumulator;
        }, []);
        const nodeIdToNeighborNodeIds = linkData.reduce((accumulator, linkDatum) => {
            if (!accumulator.hasOwnProperty(linkDatum.source)) {
                accumulator[linkDatum.source] = [];
            }
            if (!accumulator.hasOwnProperty(linkDatum.target)) {
                accumulator[linkDatum.target] = [];
            }
            accumulator[linkDatum.source].push(linkDatum.target.toString());
            accumulator[linkDatum.target].push(linkDatum.source.toString());
            return accumulator;
        }, {});
        const nodeIdToNode = nodeData.reduce((accumulator, nodeDatum) => {
            accumulator[nodeDatum.id] = nodeDatum;
            return accumulator;
        }, {});
        appendGraph(graphContainer, linkData, nodeData, nodeIdToNeighborNodeIds, nodeIdToNode);
    });

}
