import common;
import math;
import bsdf;
import medium_walk;
import surface;
import texsampler;
import sms;
import sms_transport;
import sms_work;
import sms_scene;
import sms_prior;

#define SMS_SCENE HyperionSmsScene
#define SMS_HIT HyperionSmsHit
#define SMS_TEXTURE_SAMPLER HyperionTextureSampler
#define SMS_MATERIALS materials
#define SMS_EMITTERS emissiveTriangles
#define SMS_EMITTER_CDF emissiveCdf
#include "sms_compute.slang"
