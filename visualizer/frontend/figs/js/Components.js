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
class ColoredSceneCloud extends THREE.Points {
    constructor(MAX_POINTS, size, texture = "") {
        const geometry = new THREE.BufferGeometry();
        let settings = {
            size: size,
            sizeAttenuation: false,
            alphaTest: 0.5,
            // TODO: experiment with these settings
            // transparent: false,
            // vertexColors: false,
            transparent: true,
            vertexColors: THREE.VertexColors
        };
        if (texture != "") {
            console.log(texture);
            settings["map"] = new THREE.TextureLoader().load(texture);
        }
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(MAX_POINTS * 3), 3));
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(new Float32Array(MAX_POINTS * 3), 3));
        const material = new THREE.PointsMaterial(settings);
        // material.color.set(color);

        super(geometry, material);
        this.geometry.setDrawRange(0, 0);
        
        // other class members
        this.maxPoints = MAX_POINTS;
    }

    update(points, divFactor) {
        let self = this;

        const numFeatures = 6;
        for (var i = 0; i < Math.min(points.length / numFeatures, self.maxPoints); i++) {
            // Set positions.
            const x = points[i * numFeatures + 0] / divFactor;
            const y = points[i * numFeatures + 1] / divFactor;
            const z = points[i * numFeatures + 2] / divFactor;

            self.geometry.attributes.position.array[i * 3 + 0] = x;
            self.geometry.attributes.position.array[i * 3 + 1] = y;
            self.geometry.attributes.position.array[i * 3 + 2] = z;

            // Set color.
            const r = points[i * numFeatures + 3] / 255 / divFactor;
            const g = points[i * numFeatures + 4] / 255 / divFactor;
            const b = points[i * numFeatures + 5] / 255 / divFactor;

            self.geometry.attributes.color.array[i * 3 + 0] = r;
            self.geometry.attributes.color.array[i * 3 + 1] = g;
            self.geometry.attributes.color.array[i * 3 + 2] = b;
        }
        self.geometry.setDrawRange(0, Math.min(points.length / numFeatures,
            self.maxPoints));
        self.geometry.attributes.position.needsUpdate = true;
        self.geometry.attributes.color.needsUpdate = true;
        self.geometry.computeBoundingSphere();
    }
}

class IsectPoints extends THREE.Points {
    constructor(MAX_ISECT_POINTS, POINT_SIZE, ISECT_THRESH, HEIGHT_RANGE) {
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(MAX_ISECT_POINTS * 3), 3));
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(new Float32Array(MAX_ISECT_POINTS * 3), 3));
        geometry.computeBoundingSphere();
        const material = new THREE.PointsMaterial({
            size: POINT_SIZE,
            sizeAttenuation: false,
            // alphaTest: 1,
            transparent: true,
            vertexColors: THREE.VertexColors
        });
        super(geometry, material);
        this.geometry.setDrawRange(0, 0);

        // other class members
        this.isectThresh = ISECT_THRESH;
        this.maxIsectPoints = MAX_ISECT_POINTS;
        this.heightRange = HEIGHT_RANGE;
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
            self.geometry.attributes.position.array[idx * 3 + 0] = x;
            self.geometry.attributes.position.array[idx * 3 + 1] = y;
            self.geometry.attributes.position.array[idx * 3 + 2] = z;

            // color
            const [r, g, b] = xyzi2rgb(x, y, z, i); 
            self.geometry.attributes.color.array[idx * 3 + 0] = r;
            self.geometry.attributes.color.array[idx * 3 + 1] = g;
            self.geometry.attributes.color.array[idx * 3 + 2] = b;

            idx++;
        }
        self.geometry.setDrawRange(0, Math.min(idx, self.maxIsectPoints));
        self.geometry.attributes.position.needsUpdate = true;
        self.geometry.attributes.color.needsUpdate = true;
        self.geometry.computeBoundingSphere();
    }
}

class Curtain extends THREE.Group {
    constructor(MAX_CAM_WIDTH, MAX_ISECT_POINTS, POINT_SIZE, ISECT_THRESH, HEIGHT_RANGE, COLOR_MESH, COLOR_LINE, LINE_WIDTH, OPACITY) {
        super();
        let geometry, material;
        this.maxColumns = MAX_CAM_WIDTH;
        this.maxMeshPoints = (this.maxColumns - 1) * 2 * 3;
        this.maxLinePoints = 2 * this.maxColumns + 1;
        this.y1 = -HEIGHT_RANGE[0];
        this.y2 = -HEIGHT_RANGE[1];

        // curtain mesh
        geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(this.maxMeshPoints * 3), 3));
        geometry.computeBoundingSphere();
        material = new THREE.MeshBasicMaterial({
            color: COLOR_MESH,
            opacity: OPACITY,
            transparent: true,
            depthWrite: false,  // to make transparent isect points hidden by mesh visible
            side: THREE.DoubleSide
        });
        this.curtainMesh = new THREE.Mesh(geometry, material);
        this.curtainMesh.geometry.setDrawRange(0, 0);
        this.add(this.curtainMesh);

        // curtain line
        geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(this.maxLinePoints * 3), 3));
        geometry.computeBoundingSphere();
        material = new THREE.LineBasicMaterial({
            color: COLOR_LINE,
            linewidth: LINE_WIDTH,
            opacity: 1.0,
            transparent: true,
            side: THREE.DoubleSide
        });
        this.curtainLine = new THREE.Line(geometry, material);
        this.curtainLine.geometry.setDrawRange(0, 0);
        this.add(this.curtainLine);

        // curtain intersection points
        this.isectPoints = new IsectPoints(MAX_ISECT_POINTS, POINT_SIZE, ISECT_THRESH, HEIGHT_RANGE);
        this.add(this.isectPoints);
    }

    xyzi2rgb(x, y, z, i) { return [0.0, i, 0.0]; }

    update(curtainProfile, isectPoints) {
        const self = this;
        let k = 0, idx = 0;

        // update curtain mesh
        const numColumns = Math.min(curtainProfile.length / 2, self.maxColumns);
        for (k = 0, idx = 0; k < numColumns - 1; k++) {
            const x1 = curtainProfile[2 * k + 0];
            const z1 = curtainProfile[2 * k + 1];
            const x2 = curtainProfile[2 * k + 2];
            const z2 = curtainProfile[2 * k + 3];

            const p1 = [x1, self.y1, z1];
            const p2 = [x1, self.y2, z1];
            const p3 = [x2, self.y1, z2];
            const p4 = [x2, self.y2, z2];

            // triangle 1: (p1, p2, p4)
            for (const p of [p1, p2, p4])
                for (let t = 0; t < 3; t++)
                    self.curtainMesh.geometry.attributes.position.array[idx++] = p[t];

            // triangle 2: (p1, p3, p4)
            for (const p of [p1, p3, p4])
                for (let t = 0; t < 3; t++)
                    self.curtainMesh.geometry.attributes.position.array[idx++] = p[t];
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
            const x = curtainProfile[2 * k + 0];
            const z = curtainProfile[2 * k + 1];
            const p = [x, self.y1, z];

            // line 1: (p1,)
            for (let t = 0; t < 3; t++)
                self.curtainLine.geometry.attributes.position.array[idx++] = p[t];
        }
        // pass from right to left: bottom points
        for (k = numColumns - 1; k >= 0; k--) {
            const x = curtainProfile[2 * k + 0];
            const z = curtainProfile[2 * k + 1];
            const p = [x, self.y2, z];

            // line 2: (p2,)
            for (let t = 0; t < 3; t++)
                self.curtainLine.geometry.attributes.position.array[idx++] = p[t];
        }
        // add p1 again to complete loop
        {
            const x = curtainProfile[0];
            const z = curtainProfile[1];
            const p = [x, self.y1, z];

            for (let t = 0; t < 3; t++)
                self.curtainLine.geometry.attributes.position.array[idx++] = p[t];
        }

        self.curtainLine.geometry.setDrawRange(0, Math.min(idx / 3, self.maxLinePoints));
        self.curtainLine.geometry.attributes.position.needsUpdate = true;
        self.curtainLine.geometry.computeBoundingSphere();

        // update intersection points
        self.isectPoints.update(isectPoints, self.xyzi2rgb);
    }
}
