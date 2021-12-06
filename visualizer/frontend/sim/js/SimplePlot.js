function scatter(MAX_POINTS, size, texture = "") {
    let geometry = new THREE.BufferGeometry();
    let settings = {
        size: size,
        sizeAttenuation: false,
        alphaTest: 0.5,
        transparent: true,
        vertexColors: THREE.VertexColors
    };
    if (texture != "") {
        console.log(texture);
        settings["map"] = new THREE.TextureLoader().load(texture);
    }
    geometry.addAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(MAX_POINTS * 3), 3));
    geometry.addAttribute('color', new THREE.Float32BufferAttribute(new Float32Array(MAX_POINTS * 3), 3));
    material = new THREE.PointsMaterial(settings);
    //     material.color.set(color);

    return new THREE.Points(geometry, material);
}

function scatterlcCloud(MAX_POINTS, size) {
    let geometry = new THREE.BufferGeometry();
    // let positions = [];
    // let colors = [];

    // for (var i = 0; i < points_arr.length / 4; ++i) {
    //     let x = points_arr[4 * i];
    //     let y = points_arr[4 * i + 1];
    //     let z = points_arr[4 * i + 2];
    //     let intensity = points_arr[4 * i + 3];

    //     if (enableInt16){
    //         x /= int16Factor;
    //         y /= int16Factor;
    //         z /= int16Factor;
    //         intensity /= int16Factor;   
    //     }
        
    //     positions.push(x, y, z);

    //     colors.push(0, intensity / 255, 0.2);
    // }

    // geometry.addAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    // geometry.addAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

    geometry.addAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(MAX_POINTS * 3), 3));
    geometry.addAttribute('color', new THREE.Float32BufferAttribute(new Float32Array(MAX_POINTS * 3), 3));
    geometry.computeBoundingSphere();

    let settings = {
        size: size,
        sizeAttenuation: false,
        alphaTest: 1.0,
        transparent: false,
        vertexColors: THREE.VertexColors
        // map: new THREE.TextureLoader().load("textures/sprites/disc.png")
    };

    let material = new THREE.PointsMaterial(settings);
    // let material = new THREE.PointsMaterial({size: 0.05, vertexColors: THREE.VertexColors});
    
    return new THREE.Points(geometry, material);
}

const Curtain = function(MAX_CAM_WIDTH, MAX_ISECT_POINTS, POINT_SIZE) {
    let geometry, material;
    this.isectThresh = 0.4;
    this.maxColumns = MAX_CAM_WIDTH;
    this.maxMeshPoints = (this.maxColumns - 1) * 2 * 3;
    this.maxLinePoints = 2 * this.maxColumns + 1;
    this.maxIsectPoints = MAX_ISECT_POINTS;

    // curtain mesh
    geometry = new THREE.BufferGeometry();
    geometry.addAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(this.maxMeshPoints * 3), 3));
    geometry.computeBoundingSphere();
    material = new THREE.MeshBasicMaterial({
        color: "rgb(0, 0, 0)",
        opacity: 0.4,
        transparent: true,
        side: THREE.DoubleSide
    })
    this.curtainMesh = new THREE.Mesh(geometry, material);
    this.curtainMesh.geometry.setDrawRange(0, 0);
    scene.add(this.curtainMesh);

    // curtain line
    geometry = new THREE.BufferGeometry();
    geometry.addAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(this.maxLinePoints * 3), 3));
    geometry.computeBoundingSphere();
    material = new THREE.LineBasicMaterial({
        color: "rgb(0, 0, 0)",
        linewidth: 4,
        opacity: 1.0,
        transparent: false,
        side: THREE.DoubleSide
    })
    this.curtainLine = new THREE.Line(geometry, material);
    this.curtainLine.geometry.setDrawRange(0, 0);
    scene.add(this.curtainLine);

    // curtain intersection points
    geometry = new THREE.BufferGeometry();
    geometry.addAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(this.maxIsectPoints * 3), 3));
    geometry.addAttribute(   'color', new THREE.Float32BufferAttribute(new Float32Array(this.maxIsectPoints * 3), 3));
    geometry.computeBoundingSphere();
    material = new THREE.PointsMaterial({
        size: POINT_SIZE,
        sizeAttenuation: false,
        alphaTest: 1.0,
        transparent: false,
        vertexColors: THREE.VertexColors
    });
    this.isectPoints = new THREE.Points(geometry, material);
    this.isectPoints.geometry.setDrawRange(0, 0);
    scene.add(this.isectPoints);
}

Curtain.prototype = {
    update: function (curtainBoundary, isectPoints) {
        const self = this;
        let k = 0, idx = 0, r = 0.0, b = 0.0;

        // update curtain mesh
        const numColumns = Math.min(curtainBoundary.length / 6, self.maxColumns);
        for (k = 0, idx = 0; k < numColumns - 1; k++) {
            let p1Idx = 6 * k + 0;
            let p2Idx = 6 * k + 3;
            let p3Idx = 6 * k + 6;
            let p4Idx = 6 * k + 9;

            // triangle 1: (p1, p2, p4)
            for (let t = 0; t < 3; t++)
                self.curtainMesh.geometry.attributes.position.array[idx++] = curtainBoundary[p1Idx + t];
            for (let t = 0; t < 3; t++)
                self.curtainMesh.geometry.attributes.position.array[idx++] = curtainBoundary[p2Idx + t];
            for (let t = 0; t < 3; t++)
                self.curtainMesh.geometry.attributes.position.array[idx++] = curtainBoundary[p4Idx + t];

            // triangle 2: (p1, p3, p4)
            for (let t = 0; t < 3; t++)
                self.curtainMesh.geometry.attributes.position.array[idx++] = curtainBoundary[p1Idx + t];
            for (let t = 0; t < 3; t++)
                self.curtainMesh.geometry.attributes.position.array[idx++] = curtainBoundary[p3Idx + t];
            for (let t = 0; t < 3; t++)
                self.curtainMesh.geometry.attributes.position.array[idx++] = curtainBoundary[p4Idx + t];
        }

        // TODO: see if changing this to indexed BufferGeometry speeds things up
        self.curtainMesh.geometry.setDrawRange(0, Math.min(idx / 3, self.maxMeshPoints));
        self.curtainMesh.geometry.attributes.position.needsUpdate = true;
        self.curtainMesh.geometry.computeBoundingSphere();
        // if ((color != null) && (color != self.color)) {
        //     self.color = color;
        //     self.curtainMesh.material.color.setStyle(self.color);
        //     self.curtainMesh.material.color.needsUpdate = true;
        // }

        // update curtain line
        // pass from left to right: top points
        for (k = 0, idx = 0; k < numColumns; k++) {
            let p1Idx = 6 * k + 0;

            // line 1: (p1,)
            for (let t = 0; t < 3; t++)
                self.curtainLine.geometry.attributes.position.array[idx++] = curtainBoundary[p1Idx + t];
        }
        // pass from right to left: bottom points
        for (k = numColumns - 1; k >= 0; k--) {
            let p2Idx = 6 * k + 3;

            // line 2: (p2,)
            for (let t = 0; t < 3; t++)
                self.curtainLine.geometry.attributes.position.array[idx++] = curtainBoundary[p2Idx + t];
        }
        // add p1 again to complete loop
        {
            let p1Idx = 0;
            for (let t = 0; t < 3; t++)
                self.curtainLine.geometry.attributes.position.array[idx++] = curtainBoundary[p1Idx + t];
        }

        self.curtainLine.geometry.setDrawRange(0, Math.min(idx / 3, self.maxLinePoints));
        self.curtainLine.geometry.attributes.position.needsUpdate = true;
        self.curtainLine.geometry.computeBoundingSphere();

        // update intersection points
        idx = 0;
        for (let k = 0; k < Math.min(isectPoints.length / 4, self.maxIsectPoints); k++) {
            const x = isectPoints[k * 4 + 0];
            const y = isectPoints[k * 4 + 1];
            const z = isectPoints[k * 4 + 2];
            const i = isectPoints[k * 4 + 3];

            if (i <= self.isectThresh) continue;

            // position
            self.isectPoints.geometry.attributes.position.array[idx * 3 + 0] = x;
            self.isectPoints.geometry.attributes.position.array[idx * 3 + 1] = y;
            self.isectPoints.geometry.attributes.position.array[idx * 3 + 2] = z;

            // color
            self.isectPoints.geometry.attributes.color.array[idx * 3 + 0] = r;
            self.isectPoints.geometry.attributes.color.array[idx * 3 + 1] = i;
            self.isectPoints.geometry.attributes.color.array[idx * 3 + 2] = b;

            idx++;
        }
        self.isectPoints.geometry.setDrawRange(0, Math.min(idx, self.maxIsectPoints));
        self.isectPoints.geometry.attributes.position.needsUpdate = true;
        self.isectPoints.geometry.attributes.color.needsUpdate = true;
        self.isectPoints.geometry.computeBoundingSphere();
    }
}

const SafetyEnvelope = function(numCameraRays) {
    this.numCameraRays = numCameraRays;

    // create planes
    this.planes = [];
    const material = new THREE.MeshBasicMaterial({
        color: "rgb(100, 0, 0)",
        opacity: 0.4,
        transparent: true,
        side: THREE.DoubleSide
    });
    for (let i = 0; i < this.numCameraRays - 1; i++) {
        const geometry = new THREE.PlaneGeometry(1, 6);  // height is 6m
        const plane = new THREE.Mesh(geometry, material);
        this.planes.push(plane);
        plane.visible = false;
        scene.add(plane);
    }

    this.x = new Float32Array(this.numCameraRays);
    this.z = new Float32Array(this.numCameraRays);
}

SafetyEnvelope.prototype = {

    update: function (se_points) {
        const self = this;

        console.assert(se_points.length == 2 * self.numCameraRays);
        for (let i = 0; i < self.numCameraRays; i++) {
            self.x[i] = se_points[2 * i + 0];
            self.z[i] = se_points[2 * i + 1];
        }

        // update planes of the safety envelope
        for (let i = 0; i < self.numCameraRays - 1; i++) {
            const x_curr = self.x[i], x_next = self.x[i + 1];
            const z_curr = self.z[i], z_next = self.z[i + 1];

            const x_mid = 0.5 * (x_curr + x_next);
            const z_mid = 0.5 * (z_curr + z_next);

            const width = Math.sqrt(Math.pow(x_curr - x_next, 2) + Math.pow(z_curr - z_next, 2));
            const angle = Math.atan2(x_next - x_curr, z_next - z_curr);

            const plane = self.planes[i];
            if (width < 1e-6) {
                // width is too small. scaling by a very small width creates warnings.
                // just make the plane invisible instead.
                plane.visible = false;
                continue;
            }
            plane.position.set(x_mid, 0, z_mid);
            plane.scale.set(width, 1, 1);  // expand along camera's x-axis
            plane.rotation.set(0, Math.PI/2 + angle, 0);  // rotate along camera's y axis
            plane.visible = true;
        }
    },
    invisible: function () {
        const self = this;
        for (const plane of self.planes)
            plane.visible = false;
    }
}

function scattersbCloud(MAX_POINTS) {
    let geometry = new THREE.BufferGeometry();
    let settings = {
        // size: 12,
        size: 5,
        sizeAttenuation: false,
        alphaTest: 1.0,
        transparent: false,
        color: new THREE.Color(1, 0, 0)
        // vertexColors: THREE.VertexColors
        // map: new THREE.TextureLoader().load("textures/sprites/disc.png")
    };

    geometry.addAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(MAX_POINTS * 3), 3));
    geometry.computeBoundingSphere();
    material = new THREE.PointsMaterial(settings);
    material.color.set(new THREE.Color(0.4, 0, 0));

    return new THREE.Points(geometry, material);
}

function createHeatmapPlane(image_dataurl="") {
    // image_dataurl is a string of image bytes.
    var geometry = new THREE.PlaneGeometry( 70.4, 80, 1 );
    var material;
    material = new THREE.MeshBasicMaterial({ map: dummyTexture});
    if (image_dataurl == "") {
        // For a plane with some color.
        // material = new THREE.MeshBasicMaterial( { color: 0x011B39, side: THREE.SingleSide} );
        var dummyTexture = new THREE.DataTexture( new THREE.Color( 0xffffff ), 1, 1 );
        material = new THREE.MeshBasicMaterial({ map: dummyTexture });
    }
    else {
        var texture = new THREE.TextureLoader().load(image_dataurl);
        material = new THREE.MeshBasicMaterial({ map: texture });
    }
    var plane = new THREE.Mesh( geometry, material );
    plane.position.set(70.4 / 2, 0, -3);
    return plane;
}

// function heatmapParticles() {
//     var PARTICLE_SIZE = 0.1;
//     var geometry = new THREE.BufferGeometry();
//     var material = new THREE.PointsMaterial({
//         size: PARTICLE_SIZE,
//         vertexColors: THREE.VertexColors
//     });

//     var rowNumber = 100, columnNumber = 10000;
//     var xs = [], ys = [];
//     for (var i = 0; i < 176; i++) { xs.push(0.0 + i * (70.4 / 176)) }
//     for (var i = 0; i < 200; i++) { ys.push(-40 + i * (80.0 / 200)) }
//     var particleNumber = xs.length * ys.length;
//     var positions = new Float32Array(particleNumber * 3);
//     var colors = new Float32Array(particleNumber * 3);
//     for (x_index = 0; x_index < 176; ++x_index) {
//         for (y_index = 0; y_index < 200; ++y_index) {
//             var index = (x_index * 200 + y_index) * 3;

//             // put vertices on the XY plane
//             positions[index] = xs[x_index];
//             positions[index + 1] = ys[y_index];
//             positions[index + 2] = -3;

//             // just use random color for now
//             // colors[index] = Math.random();
//             // colors[index + 1] = Math.random();
//             // colors[index + 2] = Math.random();
//             colors[index] = colors[index + 1] = colors[index + 2] = 0.5;
//         }
//     }

//     // these attributes will be used to render the particles
//     geometry.addAttribute('position', new THREE.BufferAttribute(positions, 3));
//     geometry.addAttribute('color', new THREE.BufferAttribute(colors, 3));
//     var particles = new THREE.Points(geometry, material);
//     return particles;
//     // scene.add(particles);
// }

function boxEdge(dims, pos, rots, edgewidth, color) {
    let boxes = [];
    for (var i = 0; i < dims.length; ++i) {
        let cube = new THREE.BoxGeometry(dims[i][0], dims[i][1], dims[i][2]);
        var edgeGeo = new THREE.EdgesGeometry(cube);
        let material = new THREE.LineBasicMaterial({
            color: color,
            linewidth: edgewidth
        });
        let edges = new THREE.LineSegments(edgeGeo, material);
        edges.position.set(pos[i][0], pos[i][1], pos[i][2]);
        edges.rotation.set(rots[i][0], rots[i][1], rots[i][2]);
        boxes.push(edges);
    }
    return boxes;
}

function boxEdgeWithLabel(dims, locs, rots, edgewidth, color, labels, lcolor) {
    let boxes = [];
    for (var i = 0; i < dims.length; ++i) {
        let cube = new THREE.BoxGeometry(dims[i][0], dims[i][1], dims[i][2]);
        var edgeGeo = new THREE.EdgesGeometry(cube);
        let material = new THREE.LineBasicMaterial({
            color: color,
            linewidth: edgewidth
        });
        let edges = new THREE.LineSegments(edgeGeo, material);
        edges.position.set(locs[i][0], locs[i][1], locs[i][2]);
        edges.rotation.set(rots[i][0], rots[i][1], rots[i][2]);

        var labelDiv = document.createElement( 'div' );
        labelDiv.className = 'label';
        labelDiv.textContent = labels[i];
        labelDiv.style.color = lcolor;
        // labelDiv.style.marginTop = '-1em';
        labelDiv.style.fontSize = "150%";
        // labelDiv.style.fontSize = "500%";
        var labelObj = new THREE.CSS2DObject( labelDiv );
        labelObj.position.set( 0, 0, 2 + dims[i][2]/2+locs[i][2] );
        edges.add(labelObj);
        boxes.push(edges);
    }
    return boxes;
}

function boxEdgeWithLabelV2(dims, locs, rots, edgewidth, color, labels, lcolor) {
    let boxes = [];
    for (var i = 0; i < dims.length; ++i) {
        let cube = new THREE.BoxGeometry(dims[i][0], dims[i][1], dims[i][2]);
        var edgeGeo = new THREE.EdgesGeometry(cube);
        let material = new THREE.LineBasicMaterial({
            color: color,
            linewidth: edgewidth
        });
        let edges = new THREE.LineSegments(edgeGeo, material);
        edges.position.set(locs[i][0], locs[i][1], locs[i][2]);
        edges.rotation.set(rots[i][0], rots[i][1], rots[i][2]);
        let labelObj = makeTextSprite(labels[i], {
            fontcolor: lcolor
        });
        labelObj.position.set(0, 0, dims[i][2] / 2);
        // labelObj.position.normalize();
        labelObj.scale.set(2, 1, 1.0);
        edges.add(labelObj);
        boxes.push(edges);
    }
    return boxes;
}

function box3D(dims, pos, rots, color, alpha) {
    let boxes = [];
    for (var i = 0; i < dims.length; ++i) {
        let cube = new THREE.BoxGeometry(dims[i][0], dims[i][1], dims[i][2]);
        let material = new THREE.MeshBasicMaterial({
            color: color,
            transparent: alpha != 1.0,
            opacity: alpha
        });
        let box = new THREE.Mesh(cube, material);
        box.position.set(pos[i][0], pos[i][1], pos[i][2]);
        boxes.push(box);
    }
    return boxes;
}

function makeArrows(tails, heads) {
    var arrows = [];
    for (var i = 0; i < tails.length; i += 3) {
        var tail = new THREE.Vector3(tails[i], tails[i+1], tails[i+2]);
        var dir = new THREE.Vector3(heads[i]-tails[i],
                                    heads[i+1]-tails[i+1],
                                    heads[i+2]-tails[i+2]);
        var length = dir.length();
        dir.normalize();
        var hex = 0xffff00;
        var arrow = new THREE.ArrowHelper(dir, tail, length, hex);
        arrows.push(arrow);
    }
    return arrows;
}

function getKittiInfo(backend, root_path, info_path, callback) {
    backendurl = backend + '/api/readinfo';
    data = {};
    data["root_path"] = root_path;
    data["info_path"] = info_path;
    return $.ajax({
        url: backendurl,
        method: 'POST',
        contentType: "application/json",
        data: JSON.stringify(data),
        success: function (response) {
            return callback(response["results"][0]);
        }
    });
}

function loadKittiDets(backend, det_path, callback) {
    backendurl = backend + '/api/read_detection';
    data = {};
    data["det_path"] = det_path;
    return $.ajax({
        url: backendurl,
        method: 'POST',
        contentType: "application/json",
        data: JSON.stringify(data),
        success: function (response) {
            return callback(response["results"][0]);
        }
    });
}

function getPointCloud(backend, image_idx, with_det, callback) {
    backendurl = backend + '/api/get_pointcloud';
    data = {};
    data["image_idx"] = image_idx;
    data["with_det"] = with_det;
    return $.ajax({
        url: backendurl,
        method: 'POST',
        contentType: "application/json",
        data: JSON.stringify(data),
        success: function (response) {
            return callback(response["results"][0]);
        }
    });
}

function str2buffer(str) {
    var buf = new ArrayBuffer(str.length); // 2 bytes for each char
    var bufView = new Uint8Array(buf);
    for (var i = 0, strLen = str.length; i < strLen; i++) {
        bufView[i] = str.charCodeAt(i);
    }
    return buf;
}

function choose(choices) {
    var index = Math.floor(Math.random() * choices.length);
    return choices[index];
}

function makeTextSprite(message, opts) {
    var parameters = opts || {};
    var fontface = parameters.fontface || 'Helvetica';
    var fontsize = parameters.fontsize || 70;
    var fontcolor = parameters.fontcolor || 'rgba(0, 1, 0, 1.0)';
    var canvas = document.createElement('canvas');
    var context = canvas.getContext('2d');
    context.font = fontsize + "px " + fontface;
  
    // get size data (height depends only on font size)
    var metrics = context.measureText(message);
    var textWidth = metrics.width;
  
    // text color
    context.fillStyle = fontcolor;
    context.fillText(message, 0, fontsize);
  
    // canvas contents will be used for a texture
    var texture = new THREE.Texture(canvas)
    texture.minFilter = THREE.LinearFilter;
    texture.needsUpdate = true;
  
    var spriteMaterial = new THREE.SpriteMaterial({
        map: texture,
    });
    var sprite = new THREE.Sprite(spriteMaterial);
    // sprite.scale.set(5, 5, 1.0);
    return sprite;
  }
  