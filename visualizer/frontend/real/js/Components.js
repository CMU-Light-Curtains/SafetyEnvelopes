// =====================================================================================================================
// #region Utils
// =====================================================================================================================

function addhttp(url) {
    if (!/^https?:\/\//i.test(url)) {
        url = 'http://' + url;
    }
    return url;
}

function str2buffer(str) {
    var buf = new ArrayBuffer(str.length); // 2 bytes for each char
    var bufView = new Uint8Array(buf);
    for (var i = 0, strLen = str.length; i < strLen; i++) {
        bufView[i] = str.charCodeAt(i);
    }
    return buf;
}

// #endregion
// =====================================================================================================================
class SceneCloud {
    constructor(MAX_POINTS, size, color, texture = "") {
        this.maxPoints = MAX_POINTS;

        const geometry = new THREE.BufferGeometry();
        let settings = {
            size: size,
            sizeAttenuation: false,
            alphaTest: 0.5,
            transparent: false,
            vertexColors: false,
        };
        if (texture != "") {
            console.log(texture);
            settings["map"] = new THREE.TextureLoader().load(texture);
        }
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(this.maxPoints * 3), 3));
        // Add color attribute if the color of each point could be different.
        // geometry.setAttribute('color', new THREE.Float32BufferAttribute(new Float32Array(this.maxPoints * 3), 3));
        const material = new THREE.PointsMaterial(settings);
        material.color.set(color);

        this.cloud = new THREE.Points(geometry, material);
        this.cloud.geometry.setDrawRange(0, 0);
        scene.add(this.cloud);
    }

    update(points, divFactor) {
        let self = this;

        // Set positions.
        const numFeatures = 3;
        for (var i = 0; i < Math.min(points.length / numFeatures, self.maxPoints); i++) {
            const x = points[i * numFeatures + 0] / divFactor;
            const y = points[i * numFeatures + 1] / divFactor;
            const z = points[i * numFeatures + 2] / divFactor;

            self.cloud.geometry.attributes.position.array[i * 3 + 0] = x;
            self.cloud.geometry.attributes.position.array[i * 3 + 1] = y;
            self.cloud.geometry.attributes.position.array[i * 3 + 2] = z;
        }
        self.cloud.geometry.setDrawRange(0, Math.min(points.length / numFeatures,
            self.maxPoints));
        self.cloud.geometry.attributes.position.needsUpdate = true;
        self.cloud.geometry.computeBoundingSphere();
    }
}

class IsectPoints {
    constructor(MAX_ISECT_POINTS, POINT_SIZE, ISECT_THRESH, HEIGHT_RANGE) {
        this.isectThresh = ISECT_THRESH;
        this.maxIsectPoints = MAX_ISECT_POINTS;
        this.heightRange = HEIGHT_RANGE;

        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(this.maxIsectPoints * 3), 3));
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(new Float32Array(this.maxIsectPoints * 3), 3));
        geometry.computeBoundingSphere();
        const material = new THREE.PointsMaterial({
            size: POINT_SIZE,
            sizeAttenuation: false,
            // alphaTest: 1,
            transparent: true,
            vertexColors: THREE.VertexColors
        });
        this.points = new THREE.Points(geometry, material);
        this.points.geometry.setDrawRange(0, 0);
        scene.add(this.points);
    }

    update (isectPoints, xyzi2rgb) {
        let self = this;

        let idx = 0;
        for (let k = 0; k < Math.min(isectPoints.length / 4, self.maxIsectPoints); k++) {
            const x = isectPoints[k * 4 + 0];
            const y = isectPoints[k * 4 + 1];
            const z = isectPoints[k * 4 + 2];
            const i = isectPoints[k * 4 + 3];

            if (i <= self.isectThresh) continue;
            if ((-y < self.heightRange[0]) || (-y > self.heightRange[1])) continue;

            // position
            self.points.geometry.attributes.position.array[idx * 3 + 0] = x;
            self.points.geometry.attributes.position.array[idx * 3 + 1] = y;
            self.points.geometry.attributes.position.array[idx * 3 + 2] = z;

            // color
            const [r, g, b] = xyzi2rgb(x, y, z, i); 
            self.points.geometry.attributes.color.array[idx * 3 + 0] = r;
            self.points.geometry.attributes.color.array[idx * 3 + 1] = g;
            self.points.geometry.attributes.color.array[idx * 3 + 2] = b;

            idx++;
        }
        self.points.geometry.setDrawRange(0, Math.min(idx, self.maxIsectPoints));
        self.points.geometry.attributes.position.needsUpdate = true;
        self.points.geometry.attributes.color.needsUpdate = true;
        self.points.geometry.computeBoundingSphere();
    }
}

class Curtain {
    constructor(MAX_CAM_WIDTH, MAX_ISECT_POINTS, POINT_SIZE, ISECT_THRESH, HEIGHT_RANGE) {
        let geometry, material;
        this.maxColumns = MAX_CAM_WIDTH;
        this.maxMeshPoints = (this.maxColumns - 1) * 2 * 3;
        this.maxLinePoints = 2 * this.maxColumns + 1;

        // curtain mesh
        geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(this.maxMeshPoints * 3), 3));
        geometry.computeBoundingSphere();
        material = new THREE.MeshBasicMaterial({
            color: "rgb(0, 0, 0)",
            opacity: 0.4,
            transparent: true,
            depthWrite: false,  // to make transparent isect points hidden by mesh visible
            side: THREE.DoubleSide
        });
        this.curtainMesh = new THREE.Mesh(geometry, material);
        this.curtainMesh.geometry.setDrawRange(0, 0);
        scene.add(this.curtainMesh);

        // curtain line
        geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(this.maxLinePoints * 3), 3));
        geometry.computeBoundingSphere();
        material = new THREE.LineBasicMaterial({
            color: "rgb(0, 0, 0)",
            linewidth: 4,
            opacity: 1.0,
            transparent: false,
            side: THREE.DoubleSide
        });
        this.curtainLine = new THREE.Line(geometry, material);
        this.curtainLine.geometry.setDrawRange(0, 0);
        scene.add(this.curtainLine);

        // curtain intersection points
        this.isectPoints = new IsectPoints(MAX_ISECT_POINTS, POINT_SIZE, ISECT_THRESH, HEIGHT_RANGE);
    }

    xyzi2rgb(x, y, z, i) { return [0.0, i, 0.0]; }

    update(curtainBoundary, isectPoints) {
        const self = this;
        let k = 0, idx = 0;

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
        self.isectPoints.update(isectPoints, self.xyzi2rgb);
    }
}

class IsectTracks {
    constructor (NUM_TRACKS, OPOW, MAX_ISECT_POINTS, POINT_SIZE, ISECT_THRESH, HEIGHT_RANGE) {
        let self = this;
        self.numTracks = NUM_TRACKS;
        self.opow = OPOW;
        self.isectThresh = ISECT_THRESH;
        self.mode = "monochrome";
        
        self.lutRB = new THREE.Lut('rainbow', 512);
        self.lutCW = new THREE.Lut('cooltowarm', 512);

        // tracks is a list of point clouds organized in the order of decreasing recency: elements to the right are more
        // stale than the elements to the left.
        self.tracks = new Array(self.numTracks);

        for (let i = 0; i < NUM_TRACKS; i++)
            self.tracks[i] = new IsectPoints(MAX_ISECT_POINTS, POINT_SIZE, self.isectThresh, HEIGHT_RANGE);
        
        // pointer points to the least recent (most stale) point cloud.
        // the element to its immediate right would be the most recent point cloud.
        self.ptr = 0;
    }

    update(isectPoints) {
        let self = this;

        // specify the xyzi2rgb function based on the mode
        let xyzi2rgb;
        if (self.mode == "monochrome")
            xyzi2rgb = function(x, y, z, i) { return [0.0, 1.0, 0.0]; }  // pure green
        else if (self.mode == "depth") {
            // the colormap ranges between MIN_DEPTH and MAX_DEPTH
            xyzi2rgb = function(x, y, z, i) {
                const MIN_DEPTH = 3, MAX_DEPTH = 6;
                const t = (z - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH);
                const color = self.lutRB.getColor(t);
                return [color.r, color.g, color.b];
            }
        }
        else if (self.mode == "time") {
            // the colormap cycles once in "numTracks" number of frames
            xyzi2rgb = function(x, y, z, i) {
                const t = self.ptr / self.numTracks;
                const color = self.lutCW.getColor(t);
                return [color.r, color.g, color.b];
            }
        }

        self.tracks[self.ptr].update(isectPoints, xyzi2rgb);

        // update opacity and isectThresh
        for (let i = 0; i < self.numTracks; i++) {
            const opacity = 1 - i / self.numTracks;
            const ptr = (self.ptr + i) % self.numTracks;
            self.tracks[ptr].points.material.opacity = Math.pow(opacity, self.opow);
            self.tracks[ptr].isectThresh = self.isectThresh;
        }
        
        // decrement pointer
        self.ptr = (--self.ptr) >= 0 ? self.ptr : self.numTracks - 1;
    }
}
