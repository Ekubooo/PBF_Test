using System;
using UnityEngine;
using Seb.GPUSorting;
using Unity.Mathematics;
using System.Collections.Generic;
using Seb.Helpers;
using static Seb.Helpers.ComputeHelper;

namespace Seb.Fluid.Simulation
{
	public class PBF : MonoBehaviour
	{
		public event Action<PBF> SimulationInitCompleted;

		[Header("Time Step")] 
		public float normalTimeScale = 1;
		public float slowTimeScale = 0.25f;
		public float maxTimestepFPS = 60f; 
		public int iterationsPerFrame = 3;
		public int solverIteration = 3;

		[Header("Simulation Settings")] 
		public float gravity = -10;
		public float smoothingRadius = 0.05f;
		public float targetDensity = 630;
		public float pressureMultiplier = 288;
		public float nearPressureMultiplier = 2.15f;
		public float viscosityStrength = 0;
		const float MaxDeltaVel = 3.5f;
		[Range(0, 1)] public float collisionDamping = 0.95f;

		[Header("Foam Settings")] 
		public bool foamActive;
		public int maxFoamParticleCount = 1000;
		public float trappedAirSpawnRate = 70;
		public float spawnRateFadeInTime = 0.5f;
		public float spawnRateFadeStartTime = 0;
		public Vector2 trappedAirVelocityMinMax = new(5, 25);
		public Vector2 foamKineticEnergyMinMax = new(15, 80);
		public float bubbleBuoyancy = 1.5f;
		public int sprayClassifyMaxNeighbours = 5;
		public int bubbleClassifyMinNeighbours = 15;
		public float bubbleScale = 0.5f;
		public float bubbleChangeScaleSpeed = 7;

		[Header("PBF Params")] 
		public float lambdaEps = 1000f;
		public float S_corr_K = 0f;
		public float S_corr_N = 4f;
		float rho0, deltaQ;
		

		[Header("Volumetric Render Settings")] public bool renderToTex3D;
		public int densityTextureRes;

		[Header("References")] public ComputeShader compute;
		public Spawner3D spawner;

		[HideInInspector] public RenderTexture DensityMap;
		public Vector3 Scale => transform.localScale;

		// Buffers
		public ComputeBuffer foamBuffer { get; private set; }
		public ComputeBuffer foamSortTargetBuffer { get; private set; }
		public ComputeBuffer foamCountBuffer { get; private set; }
		
		public ComputeBuffer positionBuffer { get; private set; }
		public ComputeBuffer predictPositionBuffer { get; private set; }
		public ComputeBuffer velocityBuffer { get; private set; }
		public ComputeBuffer deltaPositionBuffer { get; private set; }
		public ComputeBuffer densityBuffer { get; private set; }
		public ComputeBuffer lOperatorBuffer { get; private set; }
		public ComputeBuffer debugBuffer { get; private set; }

		ComputeBuffer sortTarget_positionBuffer;
		ComputeBuffer sortTarget_velocityBuffer;
		ComputeBuffer sortTarget_predictedPositionsBuffer;

		// Kernel IDs
		private int applyAndPredictKernel;
		private int updateSpatialHashKernel;
		private int reorderKernel;
		private int reorderCopyBackKernel;
		private int calcLagrangeOperatorKernel;
		private int calcDeltaPositionKernel;
		private int updatePredictPositionKernel;
		private int updatePropertyKernel;
		private int calcViscosityKernel;
		
		// private int UpdateDensityTextureKernel;
		// private int UpdateWhiteParticlesKernel;
		// private int WhiteParticlePrepareNextFrameKernel;
		
		SpatialHash spatialHash;

		// State
		bool isPaused;
		bool pauseNextFrame;
		float smoothRadiusOld;
		float simTimer;
		bool inSlowMode;
		Spawner3D.SpawnData spawnData;
		Dictionary<ComputeBuffer, string> bufferNameLookup;

		void Start()
		{
			Debug.Log("Controls: Space = Play/Pause, Q = SlowMode, R = Reset");
			isPaused = false;

			Initialize();
		}

		void Initialize()
		{
			spawnData = spawner.GetSpawnData();
			int numParticles = spawnData.points.Length;

			spatialHash = new SpatialHash(numParticles);
			
			// Kernel ID
			applyAndPredictKernel		= compute.FindKernel("ApplyAndPredict");
			updateSpatialHashKernel		= compute.FindKernel("UpdateSpatialHash");
			reorderKernel				= compute.FindKernel("Reorder");
			reorderCopyBackKernel		= compute.FindKernel("ReorderCopyBack");
			calcLagrangeOperatorKernel	= compute.FindKernel("CalcLagrangeOperator");
			calcDeltaPositionKernel		= compute.FindKernel("CalcDeltaPosition");
			updatePredictPositionKernel	= compute.FindKernel("UpdatePredictPosition");
			updatePropertyKernel		= compute.FindKernel("UpdateProperty");
			calcViscosityKernel			= compute.FindKernel("CalculateViscosity");
			
			
			positionBuffer			= CreateStructuredBuffer<float3>(numParticles);
			predictPositionBuffer	= CreateStructuredBuffer<float3>(numParticles);
			velocityBuffer			= CreateStructuredBuffer<float3>(numParticles);
			deltaPositionBuffer		= CreateStructuredBuffer<float3>(numParticles);
			densityBuffer			= CreateStructuredBuffer<float>(numParticles);
			lOperatorBuffer			= CreateStructuredBuffer<float>(numParticles);
			
			foamBuffer				= CreateStructuredBuffer<FoamParticle>(maxFoamParticleCount);
			foamSortTargetBuffer	= CreateStructuredBuffer<FoamParticle>(maxFoamParticleCount);
			foamCountBuffer			= CreateStructuredBuffer<uint>(4096);
			debugBuffer				= CreateStructuredBuffer<float3>(numParticles);

			sortTarget_positionBuffer			= CreateStructuredBuffer<float3>(numParticles);
			sortTarget_predictedPositionsBuffer = CreateStructuredBuffer<float3>(numParticles);
			sortTarget_velocityBuffer			= CreateStructuredBuffer<float3>(numParticles);

			bufferNameLookup = new Dictionary<ComputeBuffer, string>
			{
				{ positionBuffer, "Positions" },
				{ predictPositionBuffer, "PredictedPositions" },
				{ velocityBuffer, "Velocities" },
				{ deltaPositionBuffer, "DeltaPosition" },
				{ densityBuffer, "Densities" },
				{ lOperatorBuffer, "LOperator" },
				{ spatialHash.SpatialKeys, "SpatialKeys" },
				{ spatialHash.SpatialOffsets, "SpatialOffsets" },
				{ spatialHash.SpatialIndices, "SortedIndices" },
				{ sortTarget_positionBuffer, "SortTarget_Positions" },
				{ sortTarget_predictedPositionsBuffer, "SortTarget_PredictedPositions" },
				{ sortTarget_velocityBuffer, "SortTarget_Velocities" },
				{ foamCountBuffer, "WhiteParticleCounters" },
				{ foamBuffer, "WhiteParticles" },
				{ foamSortTargetBuffer, "WhiteParticlesCompacted" },
				{ debugBuffer, "Debug" }
			};

			// Set buffer data
			SetInitialBufferData(spawnData);

			// apply and predict kernel
			SetBuffers(compute, applyAndPredictKernel, bufferNameLookup, new ComputeBuffer[]
			{
				positionBuffer,
				predictPositionBuffer,
				velocityBuffer
			});
			
			// Spatial hash kernel
			SetBuffers(compute, updateSpatialHashKernel, bufferNameLookup, new ComputeBuffer[]
			{
				predictPositionBuffer,
				spatialHash.SpatialKeys
			});

			// Reorder kernel
			SetBuffers(compute, reorderKernel, bufferNameLookup, new ComputeBuffer[]
			{
				positionBuffer,
				predictPositionBuffer,
				velocityBuffer,
				sortTarget_positionBuffer,
				sortTarget_predictedPositionsBuffer,
				sortTarget_velocityBuffer,
				spatialHash.SpatialIndices
			});

			// Reorder copyback kernel
			SetBuffers(compute, reorderCopyBackKernel, bufferNameLookup, new ComputeBuffer[]
			{
				positionBuffer,
				predictPositionBuffer,
				velocityBuffer,
				sortTarget_positionBuffer,
				sortTarget_predictedPositionsBuffer,
				sortTarget_velocityBuffer
			});
			
			// Lagrange Operator kernel
			SetBuffers(compute, calcLagrangeOperatorKernel, bufferNameLookup, new ComputeBuffer[]
			{
				predictPositionBuffer,
				densityBuffer,
				lOperatorBuffer,
				spatialHash.SpatialOffsets,
				spatialHash.SpatialKeys
			});
			
			// Delta Position kernel
			SetBuffers(compute, calcDeltaPositionKernel, bufferNameLookup, new ComputeBuffer[]
			{
				predictPositionBuffer,
				lOperatorBuffer,
				deltaPositionBuffer,
				spatialHash.SpatialKeys,
				spatialHash.SpatialOffsets
			});
			
			// Update Predict Position kernel
			SetBuffers(compute, updatePredictPositionKernel, bufferNameLookup, new ComputeBuffer[]
			{
				predictPositionBuffer,
				deltaPositionBuffer
			});
			
			// Update Property kerenl
			SetBuffers(compute, updatePropertyKernel, bufferNameLookup, new ComputeBuffer[]
			{
				positionBuffer,
				predictPositionBuffer,
				velocityBuffer
			});
			
			SetBuffers(compute, calcViscosityKernel, bufferNameLookup, new ComputeBuffer[]
			{
				velocityBuffer,
				positionBuffer,
				predictPositionBuffer,
				spatialHash.SpatialOffsets,
				spatialHash.SpatialKeys
			});

			// Render to 3d tex kernel
			/*
			 SetBuffers(compute, renderKernel, bufferNameLookup, new ComputeBuffer[]
			{
				predictedPositionsBuffer,
				densityBuffer,
				spatialHash.SpatialKeys,
				spatialHash.SpatialOffsets,
			});

			// Foam update kernel
			SetBuffers(compute, foamUpdateKernel, bufferNameLookup, new ComputeBuffer[]
			{
				foamBuffer,
				foamCountBuffer,
				predictedPositionsBuffer,
				densityBuffer,
				velocityBuffer,
				spatialHash.SpatialKeys,
				spatialHash.SpatialOffsets,
				foamSortTargetBuffer,
				//debugBuffer
			});


			// Foam reorder copyback kernel
			SetBuffers(compute, foamReorderCopyBackKernel, bufferNameLookup, new ComputeBuffer[]
			{
				foamBuffer,
				foamSortTargetBuffer,
				foamCountBuffer,
			});
			*/

			compute.SetInt("numParticles", positionBuffer.count);
			compute.SetInt("MaxWhiteParticleCount", maxFoamParticleCount);

			UpdateSmoothingConstants();

			// Run single frame of sim with deltaTime = 0 to initialize density texture
			// (so that display can work even if paused at start)
			if (renderToTex3D)
			{
				RunSimulationFrame(0);
			}

			SimulationInitCompleted?.Invoke(this);
		}

		void Update()
		{
			// Run simulation
			if (!isPaused)
			{
				float dt = 1f / (maxTimestepFPS * ActiveTimeScale);
				RunSimulationFrame(dt);
			}

			if (pauseNextFrame)
			{
				isPaused = true;
				pauseNextFrame = false;
			}

			HandleInput();
		}

		void RunSimulationFrame(float frameDeltaTime)
		{
			float subStepDeltaTime = frameDeltaTime / iterationsPerFrame;
			UpdateSettings(subStepDeltaTime, frameDeltaTime);

			// Simulation sub-steps
			for (int i = 0; i < iterationsPerFrame; i++)
			{
				simTimer += subStepDeltaTime;
				RunSimulationStep();
			}

			/*
			// Foam and spray particles
			if (foamActive)
			{
				Dispatch(compute, maxFoamParticleCount, kernelIndex: foamUpdateKernel);
				Dispatch(compute, maxFoamParticleCount, kernelIndex: foamReorderCopyBackKernel);
			}

			// 3D density map
			if (renderToTex3D)
			{
				UpdateDensityMap();
			}
			*/
		}

		/*
		 void UpdateDensityMap()
		{
			float maxAxis = Mathf.Max(transform.localScale.x, transform.localScale.y, transform.localScale.z);
			int w = Mathf.RoundToInt(transform.localScale.x / maxAxis * densityTextureRes);
			int h = Mathf.RoundToInt(transform.localScale.y / maxAxis * densityTextureRes);
			int d = Mathf.RoundToInt(transform.localScale.z / maxAxis * densityTextureRes);
			CreateRenderTexture3D(ref DensityMap, w, h, d, UnityEngine.Experimental.Rendering.GraphicsFormat.R16_SFloat, TextureWrapMode.Clamp);
			//Debug.Log(w + " " + h + "  " + d);
			compute.SetTexture(renderKernel, "DensityMap", DensityMap);
			compute.SetInts("densityMapSize", DensityMap.width, DensityMap.height, DensityMap.volumeDepth);
			Dispatch(compute, DensityMap.width, DensityMap.height, DensityMap.volumeDepth, renderKernel);
		}
		*/

		void RunSimulationStep()
		{
			Dispatch(compute, positionBuffer.count, kernelIndex: applyAndPredictKernel);
			Dispatch(compute, positionBuffer.count, kernelIndex: updateSpatialHashKernel);
			spatialHash.Run();
			Dispatch(compute, positionBuffer.count, kernelIndex: reorderKernel);
			Dispatch(compute, positionBuffer.count, kernelIndex: reorderCopyBackKernel);

			for (int i = 0; i < solverIteration; i++)
			{
				Dispatch(compute, positionBuffer.count, kernelIndex: calcLagrangeOperatorKernel);
				Dispatch(compute, positionBuffer.count, kernelIndex: calcDeltaPositionKernel);
				Dispatch(compute, positionBuffer.count, kernelIndex: updatePredictPositionKernel);
			}
			Dispatch(compute, positionBuffer.count, kernelIndex: updatePropertyKernel);
			Dispatch(compute, positionBuffer.count, kernelIndex: calcViscosityKernel);
		}

		void UpdateSmoothingConstants()
		{
			float r = smoothingRadius;
			float spikyPow2 = 15 / (2 * Mathf.PI * Mathf.Pow(r, 5));
			float spikyPow3 = 15 / (Mathf.PI * Mathf.Pow(r, 6));
			float spikyPow2Grad = 15 / (Mathf.PI * Mathf.Pow(r, 5));
			float spikyPow3Grad = 45 / (Mathf.PI * Mathf.Pow(r, 6));

			compute.SetFloat("K_SpikyPow2", spikyPow2);
			compute.SetFloat("K_SpikyPow3", spikyPow3);
			compute.SetFloat("K_SpikyPow2Grad", spikyPow2Grad);
			compute.SetFloat("K_SpikyPow3Grad", spikyPow3Grad);
			
			rho0 = targetDensity;
			deltaQ = 0.3f * smoothingRadius;
			compute.SetFloat("rho0",rho0);
			compute.SetFloat("inv_rho0",1f/rho0);
			compute.SetFloat("deltaQ",deltaQ);	
		}

		void UpdateSettings(float stepDeltaTime, float frameDeltaTime)
		{
			if (smoothingRadius != smoothRadiusOld)
			{
				smoothRadiusOld = smoothingRadius;
				UpdateSmoothingConstants();
			}

			Vector3 simBoundsSize = transform.localScale;
			Vector3 simBoundsCentre = transform.position;

			compute.SetFloat("deltaTime", stepDeltaTime);
			compute.SetFloat("whiteParticleDeltaTime", frameDeltaTime);
			compute.SetFloat("simTime", simTimer);
			compute.SetFloat("gravity", gravity);
			compute.SetFloat("collisionDamping", collisionDamping);
			compute.SetFloat("smoothingRadius", smoothingRadius);
			compute.SetFloat("targetDensity", targetDensity);
			compute.SetFloat("pressureMultiplier", pressureMultiplier);
			compute.SetFloat("nearPressureMultiplier", nearPressureMultiplier);
			compute.SetFloat("viscosityStrength", viscosityStrength);
			compute.SetVector("boundsSize", simBoundsSize);
			compute.SetVector("centre", simBoundsCentre);

			compute.SetMatrix("localToWorld", transform.localToWorldMatrix);
			compute.SetMatrix("worldToLocal", transform.worldToLocalMatrix);
			
			// PBF
			compute.SetFloat("lambdaEps", lambdaEps);
			compute.SetFloat("S_corr_K",S_corr_K);
			compute.SetFloat("S_corr_N",S_corr_N);
			
			compute.SetFloat("g_MaxDeltaVel", MaxDeltaVel);

			// Foam settings
			float fadeInT = (spawnRateFadeInTime <= 0) ? 1 : Mathf.Clamp01((simTimer - spawnRateFadeStartTime) / spawnRateFadeInTime);
			compute.SetVector("trappedAirParams", new Vector3(trappedAirSpawnRate * fadeInT * fadeInT, trappedAirVelocityMinMax.x, trappedAirVelocityMinMax.y));
			compute.SetVector("kineticEnergyParams", foamKineticEnergyMinMax);
			compute.SetFloat("bubbleBuoyancy", bubbleBuoyancy);
			compute.SetInt("sprayClassifyMaxNeighbours", sprayClassifyMaxNeighbours);
			compute.SetInt("bubbleClassifyMinNeighbours", bubbleClassifyMinNeighbours);
			compute.SetFloat("bubbleScaleChangeSpeed", bubbleChangeScaleSpeed);
			compute.SetFloat("bubbleScale", bubbleScale);
		}

		void SetInitialBufferData(Spawner3D.SpawnData spawnData)
		{
			positionBuffer.SetData(spawnData.points);
			predictPositionBuffer.SetData(spawnData.points);
			velocityBuffer.SetData(spawnData.velocities);
			// deltaPositionBuffer.SetData(spawnData.points);

			foamBuffer.SetData(new FoamParticle[foamBuffer.count]);

			debugBuffer.SetData(new float3[debugBuffer.count]);
			foamCountBuffer.SetData(new uint[foamCountBuffer.count]);
			simTimer = 0;
		}

		void HandleInput()
		{
			if (Input.GetKeyDown(KeyCode.Space))
			{
				isPaused = !isPaused;
			}

			if (Input.GetKeyDown(KeyCode.RightArrow))
			{
				isPaused = false;
				pauseNextFrame = true;
			}

			if (Input.GetKeyDown(KeyCode.R))
			{
				pauseNextFrame = true;
				SetInitialBufferData(spawnData);
				// Run single frame of sim with deltaTime = 0 to initialize density texture
				// (so that display can work even if paused at start)
				if (renderToTex3D)
				{
					RunSimulationFrame(0);
				}
			}

			if (Input.GetKeyDown(KeyCode.Q))
			{
				inSlowMode = !inSlowMode;
			}
		}

		private float ActiveTimeScale => inSlowMode ? slowTimeScale : normalTimeScale;

		void OnDestroy()
		{
			foreach (var kvp in bufferNameLookup)
			{
				Release(kvp.Key);
			}

			spatialHash.Release();
		}


		public struct FoamParticle
		{
			public float3 position;
			public float3 velocity;
			public float lifetime;
			public float scale;
		}

		void OnDrawGizmos()
		{
			// Draw Bounds
			var m = Gizmos.matrix;
			Gizmos.matrix = transform.localToWorldMatrix;
			Gizmos.color = new Color(0, 1, 0, 0.5f);
			Gizmos.DrawWireCube(Vector3.zero, Vector3.one);
			Gizmos.matrix = m;
		}
	}
}